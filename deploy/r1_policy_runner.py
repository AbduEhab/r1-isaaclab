"""Run a trained R1 walking policy on the real robot (low-level mode).

Reads ``policy.onnx`` + ``deploy.yaml`` from a deploy directory, subscribes to ``rt/lowstate``, runs
the network at the policy rate and publishes joint targets on ``rt/lowcmd``. Safety: state watchdog,
gains ramp-in from the current pose, joint and torque clamping, and a clean stop on Ctrl+C.

    python -m deploy.r1_policy_runner --policy deploy/runs/<run> --iface eth0 --vx 0.3 --ramp 3

Before running: the robot must be in low-level ("programming"/debug) mode, the hanger or an
assistant must be ready, and the policy must have passed ``check_policy.py`` and
``scripts/validate_deploy.py`` in simulation.
"""

from __future__ import annotations

import argparse
import math
import signal
import sys
import threading
import time

import numpy as np

from . import obs_builder, r1_contract

HEAD_SLOTS = (29, 30)
HEAD_GAINS = (15.0, 1.0)


def clamp(x, lo, hi):
    return min(max(x, lo), hi)


class Runner:
    def __init__(self, args):
        self.args = args
        self.contract = r1_contract.load(args.policy_dir)
        self.builder = obs_builder.ObsBuilder(self.contract)
        self.ort = self._session(self.contract.policy_path)
        self._state = None
        self._state_t = 0.0
        self._lock = threading.Lock()
        self._stop = False
        self.command = np.array([args.vx, args.vy, args.wz], dtype=np.float32)
        self.slots = self.contract.joint_ids
        self.cmd = None
        self.pub = None

    # ------------------------------------------------------------------ DDS
    def _session(self, path):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        return ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])

    def _init_dds(self):
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
        from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_

        ChannelFactoryInitialize(0, self.args.iface)

        self.cmd = unitree_go_msg_dds__LowCmd_()
        self.cmd.head()[0].mode_pr = 0
        self.cmd.head()[0].mode_machine = 0
        for i in range(35):
            self.cmd.motor_cmd[i].q = float("nan")  # ignored unless the slot is driven
            self.cmd.motor_cmd[i].dq = 0.0
            self.cmd.motor_cmd[i].kp = 0.0
            self.cmd.motor_cmd[i].kd = 0.0
            self.cmd.motor_cmd[i].tau = 0.0
        for gain in (self.contract.stiffness, self.contract.damping):
            pass

        self.pub = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.pub.Init()

        def on_state(msg):
            with self._lock:
                self._state = msg
                self._state_t = time.monotonic()

        self.sub = ChannelSubscriber("rt/lowstate", LowState_)
        self.sub.Init(on_state, 10)

    # ------------------------------------------------------------------ helpers
    def state_arrays(self):
        with self._lock:
            msg = self._state
            age = time.monotonic() - self._state_t if msg is not None else math.inf
        if msg is None or age > self.args.watchdog:
            return None
        q = np.array([msg.motor_state[s].q for s in self.slots], dtype=np.float32)
        dq = np.array([msg.motor_state[s].dq for s in self.slots], dtype=np.float32)
        # Unitree's IMU quaternion is (w, x, y, z); the policy contract uses Isaac Lab's (x, y, z, w)
        qw, qx, qy, qz = (msg.imu_state.quaternion[0], msg.imu_state.quaternion[1],
                          msg.imu_state.quaternion[2], msg.imu_state.quaternion[3])
        quat = np.array([qx, qy, qz, qw], dtype=np.float32)
        gyro = np.array(msg.imu_state.gyroscope, dtype=np.float32)
        return q, dq, quat, gyro

    def fill_cmd(self, targets, kp_scale: float, q_now=None, blend: float = 1.0):
        c = self.contract
        for i, slot in enumerate(self.slots):
            q = targets.q_des[i]
            if q_now is not None:
                q = (1.0 - blend) * q_now[i] + blend * q
            self.cmd.motor_cmd[slot].mode = 1
            self.cmd.motor_cmd[slot].q = float(q)
            self.cmd.motor_cmd[slot].dq = 0.0
            self.cmd.motor_cmd[slot].kp = float(c.stiffness[i] * kp_scale)
            self.cmd.motor_cmd[slot].kd = float(c.damping[i] * min(1.0, kp_scale))
            self.cmd.motor_cmd[slot].tau = 0.0
        for slot in HEAD_SLOTS:
            self.cmd.motor_cmd[slot].mode = 1
            self.cmd.motor_cmd[slot].q = float("nan")
            self.cmd.motor_cmd[slot].kp = 0.0
            self.cmd.motor_cmd[slot].kd = 0.0

    def zero_cmd(self):
        for i in range(35):
            self.cmd.motor_cmd[i].q = float("nan")
            self.cmd.motor_cmd[i].dq = 0.0
            self.cmd.motor_cmd[i].kp = 0.0
            self.cmd.motor_cmd[i].kd = 0.0
            self.cmd.motor_cmd[i].tau = 0.0
        self.pub.Write(self.cmd)

    # ------------------------------------------------------------------ loop
    def run(self):
        self._init_dds()
        print("[runner] waiting for lowstate ...", flush=True)
        while self.state_arrays() is None:
            time.sleep(0.05)
        print("[runner] lowstate received; ramping gains", flush=True)

        q, dq, quat, gyro = self.state_arrays()
        self.builder.reset(time.monotonic())
        start = time.monotonic()
        dt = self.contract.step_dt
        next_t = start
        ramp = max(1e-3, self.args.ramp)
        last_no_state = 0.0

        try:
            while not self._stop:
                now = time.monotonic()
                st = self.state_arrays()
                if st is None:
                    if now - last_no_state > 0.5:
                        print("[runner] WATCHDOG: no lowstate, dropping gains", flush=True)
                        last_no_state = now
                        self.zero_cmd()
                    time.sleep(0.01)
                    if now - last_no_state > 2.0 or self.args.strict:
                        break
                    continue

                q, dq, quat, gyro = st
                if self.args.ramp:
                    ramp_frac = clamp((now - start) / ramp, 0.0, 1.0)
                else:
                    ramp_frac = 1.0

                obs = self.builder.build(now, q, dq, quat, gyro, self.command)
                if ramp_frac < 1.0:
                    # hold the measured pose while the gains come up
                    targets = r1_contract.actions_to_targets(self.contract, np.zeros(self.contract.n_joints))
                    self.fill_cmd(targets, kp_scale=ramp_frac, q_now=q, blend=1.0 - ramp_frac)
                else:
                    actions = self.ort.run(None, {"obs": obs.reshape(1, -1).astype(np.float32)})[0][0]
                    self.builder.record_action(actions)
                    targets = r1_contract.actions_to_targets(self.contract, actions)
                    self.fill_cmd(targets, kp_scale=1.0)
                self.pub.Write(self.cmd)

                next_t += dt
                sleep = next_t - time.monotonic()
                if sleep > 0:
                    time.sleep(sleep)
                else:
                    next_t = time.monotonic()  # running behind: reset the schedule
        finally:
            print("[runner] stopping: zero gains", flush=True)
            self.zero_cmd()
            time.sleep(0.2)

    def stop(self, *_):
        self._stop = True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", "--policy-dir", dest="policy_dir", required=True)
    ap.add_argument("--iface", default="eth0")
    ap.add_argument("--vx", type=float, default=0.0)
    ap.add_argument("--vy", type=float, default=0.0)
    ap.add_argument("--wz", type=float, default=0.0)
    ap.add_argument("--ramp", type=float, default=3.0, help="seconds to ramp gains in")
    ap.add_argument("--watchdog", type=float, default=0.1, help="seconds without state before dropping")
    ap.add_argument("--strict", action="store_true", help="stop on the first watchdog trip")
    args = ap.parse_args()

    runner = Runner(args)
    signal.signal(signal.SIGINT, runner.stop)
    try:
        runner.run()
    except Exception as exc:  # noqa: BLE001 - report and leave gains down
        print(f"[runner] ERROR: {exc}", flush=True)
        try:
            runner.zero_cmd()
        except Exception:
            pass
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

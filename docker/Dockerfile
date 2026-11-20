# syntax=docker/dockerfile:1.7
#
# R1 Isaac Lab image: Isaac Sim 6.2 + Isaac Lab v3.0.0 + this repository + labd.
#
#   docker build -f docker/Dockerfile -t r1-isaaclab:6.2 .
#
# Base is NVIDIA's official Isaac Sim container (Linux). The Windows binary install used during
# development cannot be containerised â€” containers are Linux â€” so this image is the reproducible
# equivalent: same Isaac Sim version, same Isaac Lab tag, same task code.
#
# Isaac Lab is installed into Isaac Sim's bundled Python, which is what `labd` and the scripts use
# through `<isaaclab>/python.sh`; no separate venv to activate.

ARG ISAACSIM_IMAGE=nvcr.io/nvidia/isaac-sim:6.2.0
ARG ISAACLAB_TAG=v3.0.0

# ---------------------------------------------------------------------------- labd (Rust, static-ish)
FROM rust:1.83-bookworm AS labd-build
WORKDIR /src
COPY labd/Cargo.toml labd/Cargo.lock ./
COPY labd/src ./src
RUN cargo build --release --locked

# ---------------------------------------------------------------------------- Isaac Sim + Isaac Lab
FROM ${ISAACSIM_IMAGE}

ARG ISAACLAB_TAG
ARG ISAACSIM_ROOT=/isaac-sim
ARG ISAACLAB_PATH=/opt/IsaacLab

ENV DEBIAN_FRONTEND=noninteractive \
    OMNI_KIT_ACCEPT_EULA=YES \
    ISAACSIM_ROOT=${ISAACSIM_ROOT} \
    ISAACLAB_PATH=${ISAACLAB_PATH} \
    PYTHONUTF8=1 \
    PYTHONIOENCODING=utf-8 \
    PYTHONUNBUFFERED=1

USER root

RUN apt-get update \
 && apt-get install -y --no-install-recommends git ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 --branch "${ISAACLAB_TAG}" https://github.com/isaac-sim/IsaacLab.git "${ISAACLAB_PATH}" \
 && ln -sfn "${ISAACSIM_ROOT}" "${ISAACLAB_PATH}/_isaac_sim" \
 && find "${ISAACLAB_PATH}" -type f -name "*.sh" -exec sed -i 's/\r$//' {} + \
 && chmod 755 "${ISAACLAB_PATH}/isaaclab.sh"

# Installs the Isaac Lab packages (editable), torch, Newton, RSL-RL and the rest into Isaac Sim's
# bundled Python. Long step: several GB from download.pytorch.org and PyPI.
RUN cd "${ISAACLAB_PATH}" && ./isaaclab.sh --install newton,rl

# ---------------------------------------------------------------------------- our task and daemon
COPY . /opt/r1-isaaclab
COPY --from=labd-build /src/target/release/labd /usr/local/bin/labd

RUN mkdir -p /artifacts
VOLUME ["/artifacts"]
EXPOSE 8080

# The daemon runs the tasks with ${ISAACLAB_PATH}/python.sh and copies finished runs to /artifacts.
ENTRYPOINT ["labd"]
CMD ["--repo", "/opt/r1-isaaclab", "--isaaclab", "/opt/IsaacLab", \
     "--port", "8080", "--bind", "0.0.0.0", "--artifacts", "/artifacts"]


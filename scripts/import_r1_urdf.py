"""Import a staged R1 URDF into USD with Isaac Sim's URDF importer.

    isaaclab.bat -p scripts/import_r1_urdf.py --urdf build/urdf/R1.urdf --out assets/r1

The importer writes ``<out>/R1/R1.usda`` plus its payloads. Keep the multi-physics conversion on:
it is what writes the ``mujoco`` variant whose Mjc joint schemas Isaac Lab's Newton backend reads
(``--flat`` drops it, which makes the asset unusable for Newton).
"""

import argparse
from pathlib import Path

from isaacsim import SimulationApp  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--urdf", required=True, type=Path)
parser.add_argument("--out", required=True, type=Path)
parser.add_argument("--flat", action="store_true", help="no multi-physics conversion (drops the mujoco variant)")
args = parser.parse_args()

app = SimulationApp({"headless": True})

import omni.kit.app  # noqa: E402

ext_manager = omni.kit.app.get_app().get_extension_manager()
for ext in ("omni.scene.optimizer.core", "isaacsim.robot.schema", "isaacsim.asset.importer.urdf"):
    ext_manager.set_extension_enabled_immediate(ext, True)

from isaacsim.asset.importer.urdf import URDFImporter, URDFImporterConfig  # noqa: E402

if not args.urdf.is_file():
    raise SystemExit(f"no staged URDF at {args.urdf}; run prepare_r1_urdf.py first")

config = URDFImporterConfig()
config.urdf_path = str(args.urdf)
config.usd_path = str(args.out)
config.robot_type = "Humanoid"
config.fix_base = False  # free base: the robot stands on its own feet
config.merge_fixed_joints = True  # R1.urdf's ankle linkage and sensor frames are fixed joints
config.allow_self_collision = False
config.run_asset_transformer = True
config.run_multi_physics_conversion = not args.flat

output_usd = URDFImporter(config).import_urdf()
print(f"\n[import_r1_urdf] USD written to: {output_usd}\n")
app.update()
app.close()

"""Stage a Unitree R1 URDF for Isaac Sim's URDF importer.

    python scripts/prepare_r1_urdf.py --urdf path/to/R1.urdf --meshes path/to/meshes --out build/urdf

Copies the URDF and every mesh it references into one folder, so the importer resolves the meshes
by relative path. Links whose mesh is missing are dropped from the copy (the team's R1.urdf
references a ``camera_link.STL`` that its mesh folder does not ship; the camera link is not
actuated and carries no collision geometry worth keeping).
"""

import argparse
import re
import shutil
import sys
from pathlib import Path


def drop_link_with_missing_mesh(text: str, mesh_dir: Path) -> tuple[str, list[str]]:
    """Remove <link> blocks whose meshes are absent, and joints whose child they were."""
    missing = sorted({m for m in re.findall(r'filename="meshes/([^"]+)"', text) if not (mesh_dir / m).is_file()})
    dropped = []
    for name in missing:
        link = re.search(r'<link name="([^"]+)">(?:(?!</link>).)*?' + re.escape(name) + r'(?:(?!</link>).)*?</link>',
                         text, flags=re.S)
        if link is None:
            continue
        link_name = link.group(1)
        text = text[: link.start()] + text[link.end():]
        text = re.sub(r'\s*<joint name="[^"]+"[^>]*>(?:(?!</joint>).)*?' + re.escape(f'<child link="{link_name}"/>')
                      + r'.*?</joint>', "", text, flags=re.S)
        dropped.append(link_name)
    return text, dropped


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--urdf", required=True, type=Path)
    ap.add_argument("--meshes", required=True, type=Path, help="folder holding the URDF's meshes/*.STL")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--keep-missing-links", action="store_true", help="fail instead of dropping them")
    args = ap.parse_args()

    if not args.urdf.is_file():
        raise SystemExit(f"no URDF at {args.urdf}")

    text = args.urdf.read_text()
    if args.keep_missing_links:
        missing = sorted({m for m in re.findall(r'filename="meshes/([^"]+)"', text)
                          if not (args.meshes / m).is_file()})
        if missing:
            raise SystemExit(f"missing meshes: {missing}")
        dropped = []
    else:
        text, dropped = drop_link_with_missing_mesh(text, args.meshes)
        if dropped:
            print(f"[prepare] dropped links with no mesh: {dropped}")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / args.urdf.name).write_text(text)

    mesh_dst = args.out / "meshes"
    mesh_dst.mkdir(exist_ok=True)
    names = sorted(set(re.findall(r'filename="meshes/([^"]+)"', text)))
    copied = 0
    for name in names:
        src = args.meshes / name
        if src.is_file():
            shutil.copy2(src, mesh_dst / name)
            copied += 1
    print(f"[prepare] copied {copied}/{len(names)} meshes into {mesh_dst}")

    joints = re.findall(r'<joint name="[^"]+" type="([^"]+)"', text)
    counts = {t: joints.count(t) for t in sorted(set(joints))}
    print(f"[prepare] joints: {len(joints)} {counts}")
    print(f"[prepare] staged at {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

# -*- coding: utf-8 -*-
"""
run_pipeline.py — agentic scene-to-data pipeline, one-command end-to-end
=========================================================================

  [1]   Blender scene build + IR dump   build_scene_bottle.py -> .blend + spec
  [1.5] Blender M0 render receipts      (3 fixed cameras)
  [2]   IR -> MJCF compile              compile_mjcf.py
  [3]   Physics settle self-check       test_mjcf_physics.py
  [4]   Reachability pre-check          check_reachability.py
  [5]   robosuite task reset test       task_bottle_in_tray.py --reset-test 15
  [6]   IK expert acceptance            expert_ik.py test --episodes 8   (>=80%)
  [6.5] Penetration audit               test_penetration.py
  [7]   Demo collection (HDF5+MP4)      collect_demos.py --episodes 8
  [7.5] Dataset read-back verification  verify_dataset.py

Usage:
    python run_pipeline.py                # full pipeline
    python run_pipeline.py --skip-blender # reuse existing .blend/spec
Exit 0 only if every stage passes.
"""

import argparse
import subprocess
import sys
import os

HERE = os.path.dirname(os.path.abspath(__file__))
BLENDER = r"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe"
PYTHON = sys.executable


def run(name, cmd):
    print("\n" + "=" * 66)
    print("STAGE:", name)
    print("=" * 66)
    r = subprocess.run(cmd, cwd=HERE)
    print(f"===> {name}: {'PASS' if r.returncode == 0 else 'FAIL'}")
    return r.returncode == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-blender", action="store_true")
    ap.add_argument("--skip-renders", action="store_true")
    ap.add_argument("--episodes", type=int, default=8)
    args = ap.parse_args()

    ok = True
    if not args.skip_blender:
        build_args = [BLENDER, "--background", "--factory-startup",
                      "--python", os.path.join(HERE, "..",
                                               "build_scene_bottle.py")]
        if args.skip_renders:
            build_args += ["--", "--skip-render"]
        ok &= run("1. Blender scene build + IR dump", build_args)

    ok &= run("2. MJCF compile", [PYTHON, os.path.join(HERE, "compile_mjcf.py")])
    ok &= run("3. Physics settle self-check",
              [PYTHON, os.path.join(HERE, "test_mjcf_physics.py")])
    ok &= run("4. Reachability pre-check",
              [PYTHON, os.path.join(HERE, "check_reachability.py")])
    ok &= run("5. robosuite task reset test",
              [PYTHON, os.path.join(HERE, "task_bottle_in_tray.py"),
               "--reset-test", "15"])
    ok &= run("6. IK expert acceptance (>=80%)",
              [PYTHON, os.path.join(HERE, "expert_ik.py"), "test",
               "--episodes", str(args.episodes)])
    ok &= run("6.5 Penetration audit",
              [PYTHON, os.path.join(HERE, "test_penetration.py")])
    ok &= run("7. Demo collection (HDF5 + MP4)",
              [PYTHON, os.path.join(HERE, "collect_demos.py"),
               "--episodes", str(args.episodes)])
    ok &= run("7.5 Dataset read-back verification",
              [PYTHON, os.path.join(HERE, "verify_dataset.py")])

    print("\n" + "=" * 66)
    print("PIPELINE_RESULT:", "ALL_OK" if ok else "FAILED")
    print("=" * 66)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

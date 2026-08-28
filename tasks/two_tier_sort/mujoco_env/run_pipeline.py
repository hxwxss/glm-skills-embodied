# -*- coding: utf-8 -*-
"""
run_pipeline.py -- one-command end-to-end run of every pipeline stage
=====================================================================

  [1] M0   scene build (IR dump)      build_scene.py
  [2] M1   IR -> MJCF compile         compile_mjcf.py
  [3] M1   physics self-check         test_mjcf_physics.py
                                       (settle / no penetration / hinge
                                        flip-and-hold / drawer slide-and-hold)
  [4] M1.5 reachability pre-check     check_reachability.py
  [5] M2   robosuite task check       task_two_tier.py --reset-test 10
  [6] M3   IK expert acceptance       expert_ik.py test --episodes 6  (>=80%)
  [6.5] M3.5 penetration audit        test_penetration.py
  [7] M4   demo capture               collect_demos.py --episodes 1
                                       (HDF5 + MP4, readback-verified)

Exit code 0 = every gate passed.
"""

import argparse
import subprocess
import sys
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PYTHON = sys.executable


def run(name, cmd):
    print("\n" + "=" * 66)
    print("STAGE:", name)
    print("=" * 66)
    r = subprocess.run(cmd, cwd=HERE)
    print("===> %s: %s" % (name, "PASS" if r.returncode == 0 else "FAIL"))
    return r.returncode == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=6,
                    help="expert acceptance episodes")
    ap.add_argument("--collect-episodes", type=int, default=1,
                    help="demos written to the HDF5")
    args = ap.parse_args()

    ok = True
    ok &= run("1. M0 scene build (IR)", [PYTHON, os.path.join(HERE, "..",
             "build_scene.py")])
    ok &= run("2. M1 MJCF compile", [PYTHON, "compile_mjcf.py"])
    ok &= run("3. M1 physics self-check", [PYTHON, "test_mjcf_physics.py"])
    ok &= run("4. M1.5 reachability pre-check",
              [PYTHON, "check_reachability.py"])
    ok &= run("5. M2 robosuite task check",
              [PYTHON, "task_two_tier.py", "--reset-test", "10"])
    ok &= run("6. M3 IK expert acceptance (>=80%%)",
              [PYTHON, "expert_ik.py", "test", "--episodes",
               str(args.episodes)])
    ok &= run("6.5 M3.5 penetration audit", [PYTHON, "test_penetration.py"])
    ok &= run("7. M4 demo capture (HDF5 + MP4)",
              [PYTHON, "collect_demos.py", "--episodes",
               str(args.collect_episodes)])

    print("\n" + "=" * 66)
    print("PIPELINE_RESULT:", "ALL_OK" if ok else "FAILED")
    print("=" * 66)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

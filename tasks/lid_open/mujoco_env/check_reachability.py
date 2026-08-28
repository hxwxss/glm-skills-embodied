# -*- coding: utf-8 -*-
"""
check_reachability.py -- M1.5: reachability pre-check (IK-based)
=================================================================

Cheapest possible robot-work sanity gate, run BEFORE expert tuning:
build the full env, then numerically solve IK (same solver the expert
uses) to every task-critical keypoint computed from the spec:

  * grasp pose above the knob (closed lid)
  * every arc waypoint of the pull (worst case: the far end, lid open)
  * retreat point

PASS requires every residual < 5 mm.  Also prints the horizontal
base->target radii against the Panda working band (0.15-0.78 m).

Exit 0 = PASS, 1 = FAIL.
"""

import os
import sys

import numpy as np

from expert_ik import IKExpert, make_env, GRASP_ROT

HERE = os.path.dirname(os.path.abspath(__file__))
REACH_MIN, REACH_MAX = 0.15, 0.78
TOL_M = 0.005


def main():
    env = make_env()
    ex = IKExpert(env)
    env.reset()
    ex.rebind(env)

    spec = env.spec
    ex_cfg = spec["task"]["expert"]
    dz = float(ex_cfg["grasp_dz_m"])
    base = np.array(spec["robots"][0]["base_pos"])

    head0 = env.knob_pos()
    hinge = np.array(env.sim.data.body_xpos[env.lid_body_id])

    targets = [("grasp@closed", head0 + np.array([0, 0, dz]))]
    thetas = np.linspace(float(ex_cfg["arc_start_rad"]),
                         float(ex_cfg["arc_end_rad"]), int(ex_cfg["arc_steps"]))
    for th in thetas:
        targets.append(("arc@%.2frad" % th,
                        ex.knob_world_at(th, hinge) + np.array([0, 0, dz])))
    targets.append(("retreat",
                    ex.knob_world_at(thetas[-1], hinge)
                    + np.array([0, 0, dz])
                    + np.array(ex_cfg["retreat_offset_m"])))

    failures = []
    advisories = []
    print("=" * 64)
    print("REACHABILITY PRE-CHECK: position residual < %.0f mm AND radius in "
          "(%.2f, %.2f] m;  orientation residual is advisory (the fixed "
          "top-down grasp tilts up to ~20 deg along the pull arc, which the"
          " bar grip tolerates -- validated end-to-end by the M3 gate)"
          % (TOL_M * 1e3, REACH_MIN, REACH_MAX))
    print("=" * 64)
    for name, p in targets:
        q, perr, rerr = ex.solve_ik(p)
        r = float(np.hypot(p[0] - base[0], p[1] - base[1]))
        in_band = REACH_MIN < r <= REACH_MAX
        ok = perr < TOL_M and in_band
        if rerr > 0.35:
            advisories.append("%s rot %.1f deg" % (name, np.degrees(rerr)))
        print("[%s] %-16s pos(% .3f,% .3f,% .3f)  radius=%.3f m  "
              "pos_res=%.1f mm  rot_tilt=%.1f deg"
              % ("ok" if ok else "FAIL", name, p[0], p[1], p[2], r,
                 perr * 1e3, np.degrees(rerr)))
        if perr >= TOL_M:
            failures.append("%s ik position residual %.1f mm"
                            % (name, perr * 1e3))
        if not in_band:
            failures.append("%s radius %.3f m outside working band" % (name, r))
    for a in advisories:
        print("[advisory] large wrist tilt: %s" % a)
    env.close()

    print("-" * 64)
    if failures:
        print("REACHABILITY_FAILED")
        for f in failures:
            print("  -", f)
        sys.exit(1)
    print("REACHABILITY_OK")
    sys.exit(0)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
check_reachability.py -- M1.5: reachability pre-check (IK-based)
=================================================================

Cheapest possible robot-work sanity gate, run BEFORE expert tuning:
build the full env, plan the expert's full waypoint set from a random
reset (lid arc, drawer pull, both pick-and-place phases, retreats), then
numerically solve IK (same solver the expert uses) to EVERY waypoint.

PASS requires every residual < 5 mm and the horizontal base->target radius
inside the Panda working band (0.15-0.78 m).

Exit 0 = PASS, 1 = FAIL.
"""

import os
import sys

import numpy as np

from expert_ik import IKExpert, make_env

HERE = os.path.dirname(os.path.abspath(__file__))
REACH_MIN, REACH_MAX = 0.15, 0.78
TOL_M = 0.005


def main():
    env = make_env()
    ex = IKExpert(env)
    env.reset()
    ex.rebind(env)

    base = np.array(env.spec["robots"][0]["base_pos"])
    wps = ex.plan()

    failures = []
    advisories = []
    print("=" * 64)
    print("REACHABILITY PRE-CHECK: %d waypoints; position residual < %.0f mm "
          "AND radius in (%.2f, %.2f] m (orientation residual advisory -- "
          "the fixed top-down grasp tilts along the lid arc, tolerated by "
          "the bar grip and validated end-to-end by the M3 gate)"
          % (len(wps), TOL_M * 1e3, REACH_MIN, REACH_MAX))
    print("=" * 64)
    prev_q = None
    for wp in wps:
        if wp["pos"] is None:
            continue
        wp = ex.resolve(wp)
        p = wp["pos"]
        name = wp["note"]
        q, perr, rerr = ex.solve_ik(p, seed=prev_q, rot=wp.get("rot"),
                                    seeded_only=wp.get("seeded", False))
        prev_q = q
        r = float(np.hypot(p[0] - base[0], p[1] - base[1]))
        in_band = REACH_MIN < r <= REACH_MAX
        ok = perr < TOL_M and in_band
        if rerr > 0.35:
            advisories.append("%s rot %.1f deg" % (name, np.degrees(rerr)))
        print("[%s] %-22s pos(% .3f,% .3f,% .3f)  radius=%.3f m  "
              "pos_res=%.1f mm  rot_tilt=%.1f deg"
              % ("ok" if ok else "FAIL", name, p[0], p[1], p[2], r,
                 perr * 1e3, np.degrees(rerr)))
        if perr >= TOL_M:
            failures.append("%s ik position residual %.1f mm"
                            % (name, perr * 1e3))
        if not in_band:
            failures.append("%s radius %.3f m outside working band"
                            % (name, r))
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

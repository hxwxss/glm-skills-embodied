# -*- coding: utf-8 -*-
"""check_reachability.py — M1.5 reachability pre-check.

For every task-critical object (semantic grasp_target / container), the
horizontal base->object distance must sit inside the Panda workspace
(~0.15-0.78 m). Because the IR randomizes init poses, the ENTIRE jitter
rectangle is checked: all 4 corners of each object's jitter box must be
reachable, so every random reset is guaranteed reachable.

Static decorations (distractor etc.) are exempt.
"""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SPEC = os.path.join(HERE, "..", "spec", "scene_spec.json")

REACH_R = 0.78          # Panda practical working radius (nominal 0.855)
MIN_R = 0.15            # too close = base collision


def main():
    with open(SPEC, encoding="utf-8") as fh:
        spec = json.load(fh)
    robot = spec["robots"][0]
    bx, by, _ = robot["base_pos"]
    jitter = spec["task"]["init_randomization"]
    jitter_map = {
        "grasp_target": jitter.get("grasp_object_xy_jitter_m", 0.0),
        "container": jitter.get("tray_xy_jitter_m", 0.0),
    }
    must_reach = {"grasp_target", "container"}
    fail = []
    for obj in spec["objects"]:
        sems = set(obj.get("semantic", []))
        if not (sems & must_reach):
            continue
        p = obj.get("body_pos") or obj["pos"]
        jj = 0.0
        for s in sems & must_reach:
            jj = max(jj, jitter_map.get(s, 0.0))
        # check the nominal point AND all 4 corners of the jitter rectangle
        pts = [(p[0], p[1])]
        if jj > 0:
            pts += [(p[0] + sx * jj, p[1] + sy * jj)
                    for sx in (-1, 1) for sy in (-1, 1)]
        dists = [math.hypot(qx - bx, qy - by) for qx, qy in pts]
        d_nom, d_min, d_max = dists[0], min(dists), max(dists)
        ok = MIN_R < d_min and d_max <= REACH_R
        print(f"[{'ok' if ok else 'FAIL'}] {obj['id']:12s} horizontal reach "
              f"nominal={d_nom:.3f}  jitter-rect=[{d_min:.3f}, {d_max:.3f}] m "
              f"(allowed {MIN_R}-{REACH_R})")
        if not ok:
            fail.append(obj["id"])

    if fail:
        print("REACHABILITY_FAILED:", fail)
        sys.exit(1)
    print("REACHABILITY_OK")
    sys.exit(0)


if __name__ == "__main__":
    main()

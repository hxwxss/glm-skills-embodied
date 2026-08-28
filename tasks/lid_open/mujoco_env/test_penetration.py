# -*- coding: utf-8 -*-
"""
test_penetration.py -- M3.5: penetration audit of a full expert episode
========================================================================

Logs every MuJoCo contact per control step while the expert runs, and
aggregates max penetration depth per geometry pair.

  whitelist (legitimate contact):
    fingers <-> lid_handle_bar / lid_handle_bracket   (the grasp)
    lid_panel <-> box walls                            (resting closed and
                                                        resting at open stop)
  blacklist (never allowed):
    robot arm links (non-finger geoms) <-> anything    depth > 2 mm
  thresholds:
    any pair with sustained depth > 8 mm or a spike > 25 mm

Also verifies the end state: lid open past the success threshold.

Exit 0 = PASS, 1 = FAIL.
"""

import os
import sys

import numpy as np
import mujoco

from expert_ik import IKExpert, make_env

HERE = os.path.dirname(os.path.abspath(__file__))

PEN_WARN = 0.008     # 8 mm sustained
PEN_FAIL = 0.025     # 25 mm spike
DEPTH_EPS = 1e-9

GRASP_GEOMS = {"lid_handle_bar", "lid_handle_bracket"}
BOX_GEOMS = {"box_front", "box_back", "box_left", "box_right", "box_bottom"}


def main():
    env = make_env()
    ex = IKExpert(env)
    env.reset()
    ex.rebind(env)

    m, d = env.sim.model._model, env.sim.data._data
    gname = lambda gid: (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gid)
                         or "geom%d" % gid)

    finger_geoms = set()
    arm_geoms = set()
    for g in range(m.ngeom):
        body = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY,
                                 m.geom_bodyid[g]) or ""
        if "finger" in gname(g) or "finger" in body:
            finger_geoms.add(g)
        elif body.startswith("robot0") or body.startswith("gripper0"):
            arm_geoms.add(g)

    worst = {}          # pair -> (max_depth, step)
    arm_hits = []
    n_steps = 0

    def scan():
        nonlocal n_steps
        n_steps += 1
        for i in range(d.ncon):
            c = d.contact[i]
            if c.dist >= -DEPTH_EPS:
                continue
            depth = -c.dist
            g1, g2 = c.geom1, c.geom2
            pair = tuple(sorted((gname(g1), gname(g2))))
            if depth > worst.get(pair, (0, 0))[0]:
                worst[pair] = (depth, n_steps)
            if (g1 in arm_geoms or g2 in arm_geoms) and depth > 0.002:
                other = gname(g2 if g1 in arm_geoms else g1)
                arm_hits.append((n_steps, other, round(depth, 4)))

    def step_scan(a=None):
        if a is not None:
            env.step(a)
        else:
            pass
        scan()

    # replicate expert.run() but scan after every control step
    wps = ex.plan()
    prev_q = None
    done = False
    for wp in wps:
        if done:
            break
        if wp["pos"] is not None:
            q_arm, _, _ = ex.solve_ik(wp["pos"], seed=prev_q)
            prev_q = q_arm
            held = 0
            for _ in range(400):
                qa = d.qpos[ex.arm_adrs]
                a = np.zeros(env.action_dim)
                a[:7] = qa + np.clip(q_arm - qa, -0.25, 0.25)
                a[7] = float(wp["gripper"])
                obs, _, done, _ = env.step(a)
                scan()
                if np.all(np.abs(q_arm - qa) < 0.03):
                    held += 1
                    if held >= 15:
                        break
                else:
                    held = 0
                if done:
                    break
            dwell = int(wp.get("dwell") or 0)
            if dwell and not done:
                hold = np.zeros(env.action_dim)
                hold[:7] = d.qpos[ex.arm_adrs]
                hold[7] = float(wp["gripper"])
                for _ in range(dwell):
                    obs, _, done, _ = env.step(hold)
                    scan()
                    if done:
                        break
        else:
            hold = np.zeros(env.action_dim)
            hold[:7] = d.qpos[ex.arm_adrs]
            hold[7] = float(wp["gripper"])
            for _ in range(int(wp.get("dwell") or 30)):
                obs, _, done, _ = env.step(hold)
                scan()
                if done:
                    break

    def classify(pair, depth):
        a, b = pair
        fing = lambda n: "finger" in n
        if a.startswith("gripper0_right_finger") and \
                b.startswith("gripper0_right_finger"):
            return "ok(fingers)"     # pads meet after release; 0.0 mm depth
        if (a in GRASP_GEOMS and fing(b)) or (b in GRASP_GEOMS and fing(a)):
            return "ok(grasp)"
        if (a == "lid_panel" and fing(b)) or (b == "lid_panel" and fing(a)):
            return "ok(shallow)" if depth <= 0.004 else "SUSPECT"
        lid_pair = {"lid_panel"} | GRASP_GEOMS
        if (a in lid_pair and b in BOX_GEOMS) or (b in lid_pair and
                                                  a in BOX_GEOMS):
            return "ok(rest)"
        return "SUSPECT"

    print("=" * 64)
    print("PENETRATION AUDIT (episode) control_steps=%d" % n_steps)
    print("=" * 64)
    failures = []

    if arm_hits:
        uniq = sorted(set(h[1] for h in arm_hits))
        print("[FAIL] arm link collided with: %s (%d steps)"
              % (uniq, len(arm_hits)))
        failures.append("arm link collision")
    else:
        print("[ok] A. no arm-link collisions (fingers excluded)")

    for pair, (depth, step) in sorted(worst.items(), key=lambda kv: -kv[1][0]):
        status = classify(pair, depth)
        if depth > PEN_FAIL:
            status = "HARD-TUNNEL"
            failures.append("tunnel %s %.3f m" % (str(pair), depth))
        elif depth > PEN_WARN and "ok" not in status:
            failures.append("deep penetration %s %.3f m" % (str(pair), depth))
        print("[%-11s] %-32s <-> %-32s max_pen=%.1f mm @step %d"
              % (status, pair[0][:32], pair[1][:32], depth * 1000, step))

    th = env.lid_angle()
    ok_end = th <= env.lid_angle_max
    print("[%s] end state: lid angle %.3f rad (<= %.2f required)"
          % ("ok" if ok_end else "FAIL", th, env.lid_angle_max))
    if not ok_end:
        failures.append("lid not open at end")

    print("-" * 64)
    if failures:
        print("PENETRATION_RESULT: FAILED")
        for f in failures:
            print("  -", f)
        sys.exit(1)
    print("PENETRATION_RESULT: PASS")
    sys.exit(0)


if __name__ == "__main__":
    main()

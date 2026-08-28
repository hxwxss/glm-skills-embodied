# -*- coding: utf-8 -*-
"""
test_penetration.py -- M3.5: penetration audit of a full expert episode
========================================================================

Logs every MuJoCo contact per control step while the expert runs the FULL
task (lid -> drawer -> red place -> blue place), aggregates max penetration
depth per geometry pair, then verifies the end state holds through a 1.5 s
settle (both cubes must STAY in their compartments).

  whitelist (legitimate contact):
    fingers <-> lid/drawer handle bars and brackets    (the grasps)
    fingers <-> cubes                                  (the grasps)
    lid_panel <-> upper walls                          (resting closed /
                                                        resting at open stop)
    tray <-> housing bottom / housing walls            (slide carriage)
    cubes <-> table / tray / upper walls               (resting, placed)
  blacklist (never allowed):
    robot arm links (non-finger geoms) <-> anything    depth > 2 mm
  thresholds:
    any pair with sustained depth > 8 mm or a spike > 25 mm

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

GRASP_GEOMS = {"lid_handle_bar", "lid_handle_bracket",
               "drawer_handle_bar", "drawer_handle_bracket"}
CUBE_GEOMS = {"RedCube_g0", "BlueCube_g0"}
WALL_GEOMS = {"housing_bottom", "housing_left", "housing_right",
              "housing_back", "housing_top", "upper_front", "upper_back",
              "upper_left", "upper_right"}
TRAY_GEOMS = {"tray_bottom", "tray_front", "tray_back", "tray_left",
              "tray_right"}


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

    # run the full verified-phase expert (with its retries) and scan every
    # control step
    def step_cb(action, obs):
        scan()

    succ, n_steps_run, info = ex.execute(env, step_cb=step_cb)
    n_steps = n_steps_run

    # hold-still probe: the placement must survive a 1.5 s settle
    hold = np.zeros(env.action_dim)
    hold[:7] = d.qpos[ex.arm_adrs]
    hold[7] = -1.0
    for _ in range(30):
        env.step(hold)
        scan()
    succ_after = env._check_success()

    def classify(pair, depth):
        a, b = pair
        fing = lambda n: n.startswith("gripper0_right_finger")
        if fing(a) and fing(b):
            return "ok(fingers)"
        if (a in GRASP_GEOMS and fing(b)) or (b in GRASP_GEOMS and fing(a)):
            return "ok(grasp)"
        if (a in CUBE_GEOMS and fing(b)) or (b in CUBE_GEOMS and fing(a)):
            return "ok(grasp)" if depth <= 0.006 else "SUSPECT"
        lid_set = {"lid_panel"} | GRASP_GEOMS
        if (a in lid_set and b in WALL_GEOMS) or (b in lid_set and
                                                  a in WALL_GEOMS):
            return "ok(rest)"
        tray_set = TRAY_GEOMS | GRASP_GEOMS
        if (a in tray_set and b in WALL_GEOMS) or (b in tray_set and
                                                   a in WALL_GEOMS):
            return "ok(slide)"
        if (a in CUBE_GEOMS and (b in WALL_GEOMS or b in TRAY_GEOMS)) or \
           (b in CUBE_GEOMS and (a in WALL_GEOMS or a in TRAY_GEOMS)):
            return "ok(rest)"
        if a.startswith("table") or b.startswith("table"):
            if a in CUBE_GEOMS or b in CUBE_GEOMS:
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
        print("[%-11s] %-28s <-> %-28s max_pen=%.1f mm @step %d"
              % (status, pair[0][:28], pair[1][:28], depth * 1000, step))

    th = env.lid_angle()
    sl = env.drawer_slide()
    lid_closed = abs(th) <= float(
        env.spec["task"]["success_condition"]["lid_closed_rad"])
    drawer_closed = sl <= float(
        env.spec["task"]["success_condition"]["drawer_closed_m"])
    print("[%s] end state: lid %.3f rad (closed), drawer %.3f m (closed)"
          % ("ok" if lid_closed and drawer_closed else "FAIL", th, sl))
    if not (lid_closed and drawer_closed):
        failures.append("articulations not closed at end")
    uc, uh = env.upper_zone_world()
    tc, th2 = env.tray_zone_world()
    red, blue = env.prop_pos("RedCube"), env.prop_pos("BlueCube")
    print("[%s] end state: red in upper zone (off=%s mm)"
          % ("ok" if np.all(np.abs(red - uc) < uh * 0.98) else "FAIL",
             np.round((red - uc) * 1000, 1).tolist()))
    print("[%s] end state: blue in tray zone (off=%s mm)"
          % ("ok" if np.all(np.abs(blue - tc) < th2 * 0.98) else "FAIL",
             np.round((blue - tc) * 1000, 1).tolist()))
    if not np.all(np.abs(red - uc) < uh * 0.98):
        failures.append("red not in upper compartment at end")
    if not np.all(np.abs(blue - tc) < th2 * 0.98):
        failures.append("blue not in drawer at end")
    print("[%s] placements survive 1.5 s settle: success=%s"
          % ("ok" if succ_after else "FAIL", succ_after))
    if not succ_after:
        failures.append("placement does not survive settle")

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

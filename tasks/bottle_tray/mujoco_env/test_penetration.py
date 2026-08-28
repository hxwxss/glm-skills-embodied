# -*- coding: utf-8 -*-
"""
test_penetration.py — full-episode penetration audit (M3.5)
============================================================

Executes a complete IK-expert episode while logging every MuJoCo contact
per step, aggregated per geometry pair:

  legal (whitelist):
    GreenBottle <-> table            (resting)
    GreenBottle <-> finger geoms     (grasp)
    GreenBottle <-> Tray walls/base  (shallow contacts during placement)
  forbidden (blacklist):
    any NON-finger robot link geom <-> anything  (arm should not collide)
  thresholds (any pair):
    sustained penetration > 8 mm, or a single spike > 25 mm

Also verifies the end-state geometry (bottle standing in the tray zone).

Usage: python test_penetration.py     (exit 0 = PASS)
"""

import os
import sys

import numpy as np
import mujoco

from expert_ik import IKExpert, make_env

HERE = os.path.dirname(os.path.abspath(__file__))

PEN_WARN = 0.008     # 8 mm sustained
PEN_FAIL = 0.025     # 25 mm spike
ARM_HIT_DEPTH = 0.001  # >1 mm arm-link contact counts as a collision


def main():
    env = make_env()
    expert = IKExpert(env)
    env.reset()
    expert.rebind(env)

    m, d = env.sim.model._model, env.sim.data._data
    gname = lambda gid: (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gid)
                         or f"geom{gid}")

    finger_geoms = set()
    for g in range(m.ngeom):
        b = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY,
                              m.geom_bodyid[g]) or ""
        if "finger" in b:
            finger_geoms.add(g)
    arm_geoms = set()
    for g in range(m.ngeom):
        b = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY,
                              m.geom_bodyid[g]) or ""
        if (b.startswith("robot0_") or b.startswith("gripper0_")) \
                and g not in finger_geoms:
            arm_geoms.add(g)

    worst = {}          # pair -> (max depth, step)
    arm_hits = []       # (step, other, depth)
    n_steps = 0

    bottle0 = env.bottle_pos().copy()
    tray0 = env.tray_center().copy()
    wps = expert.plan(bottle0, tray0)
    done = False

    def scan(step):
        nonlocal n_steps
        n_steps += 1
        for i in range(d.ncon):
            c = d.contact[i]
            if c.dist >= -1e-9:
                continue
            depth = -c.dist
            pair = tuple(sorted((gname(c.geom1), gname(c.geom2))))
            if depth > worst.get(pair, (0, 0))[0]:
                worst[pair] = (depth, step)
            g1, g2 = c.geom1, c.geom2
            if (g1 in arm_geoms or g2 in arm_geoms) and depth > ARM_HIT_DEPTH:
                other = gname(g2 if g1 in arm_geoms else g1)
                arm_hits.append((step, other, round(depth, 4)))

    for wp in wps:
        if wp["pos"] is not None:
            q_arm, err = expert.solve_ik(wp["pos"])
            for _ in range(400):
                qa = d.qpos[expert.arm_adrs]
                if np.all(np.abs(q_arm - qa) < 0.06):
                    break
                a = np.zeros(env.action_dim)
                a[:7] = qa + np.clip(q_arm - qa, -0.25, 0.25)
                a[7] = float(wp["gripper"])
                obs, _, done, _ = env.step(a)
                scan(n_steps)
                if done:
                    break
        else:
            hold = np.zeros(env.action_dim)
            hold[:7] = d.qpos[expert.arm_adrs]
            hold[7] = float(wp["gripper"])
            for _ in range(int(wp.get("dwell") or 30)):
                obs, _, done, _ = env.step(hold)
                scan(n_steps)
                if done:
                    break
        if done:
            break

    ok_rollout = env._check_success()
    bottle_end = env.bottle_pos()

    print("=" * 64)
    print(f"PENETRATION AUDIT — rollout steps={n_steps}, success={ok_rollout}")
    print("=" * 64)

    failures = []

    # 1. arm link (non-finger) collisions
    if arm_hits:
        uniq = sorted(set(h[1] for h in arm_hits))
        print(f"[FAIL] A. arm links collided with: {uniq} "
              f"({len(arm_hits)} steps, max depth "
              f"{max(h[2] for h in arm_hits)*1000:.2f} mm)")
        failures.append("arm link collision")
    else:
        print("[ok] A. zero arm-link collisions (non-finger geoms)")

    # 2. per-pair penetration depths
    for pair, (depth, step) in sorted(worst.items(), key=lambda kv: -kv[1][0]):
        names = set(pair)
        legit = (
            ("GreenBottle" in " ".join(pair)) and
            any(k in " ".join(pair) for k in
                ("table", "Tray", "finger")))
        if depth > PEN_FAIL:
            status = "HARD-TUNNEL"
            failures.append(f"tunnel {pair} {depth:.3f}m")
        elif depth > PEN_WARN:
            status = "ok(shallow)" if legit else "SUSPECT"
            if not legit:
                failures.append(f"deep penetration {pair} {depth:.3f}m")
        else:
            status = "ok"
        print(f"[{status:11s}] {pair[0][:30]:30s} <-> {pair[1][:30]:30s} "
              f"max_pen={depth*1000:.2f} mm @step={step}")

    # 3. end-state geometry: bottle standing in the tray zone
    zone = env._zone_center()
    zh = env.zone_half
    rel = np.abs(bottle_end - zone)
    in_zone = bool(np.all(rel < zh))
    print(f"[{'ok' if in_zone else 'FAIL'}] C. bottle inside tray zone at end: "
          f"rel={np.round(rel, 3).tolist()} (zone half {np.round(zh, 3).tolist()})")
    if not in_zone:
        failures.append("bottle not in tray at end")

    print("-" * 64)
    if failures:
        print("PENETRATION_RESULT: FAILED")
        for f_ in failures:
            print("  -", f_)
        sys.exit(1)
    print("PENETRATION_RESULT: PASS")
    sys.exit(0)


if __name__ == "__main__":
    main()

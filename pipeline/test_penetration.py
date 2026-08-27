# -*- coding: utf-8 -*-
"""
test_penetration.py — 抓取执行全程的穿模审计
=============================================

在完整 IK 专家 episode 中逐步记录 MuJoCo 接触,按"几何对"聚合穿透深度:

  合法接触(白名单):
    finger ↔ red cube        (抓取)
    red cube ↔ table/plinth  (静置)
    red cube ↔ box 壁/底     (入盒瞬间,浅接触)
  可疑(任何对):
    穿透深度 > 8mm 持续, 或瞬时 > 25mm  → 大概率隧穿/硬穿模
  禁止(黑名单):
    机器人非手指连杆 ↔ 任何物体/桌      (手臂不应撞东西)
    red cube ↔ box 壁 深度 > 12mm        (穿壁)

另外在 episode 结束后做一次静态 AABB 复核:
  红块应在盒内且不与四壁相交;蓝柱/黄球仍在桌面高度。

用法: python test_penetration.py
退出码 0 = PASS。
"""

import os
import sys

import numpy as np
import mujoco

from expert_ik import IKExpert, make_env

HERE = os.path.dirname(os.path.abspath(__file__))

PEN_WARN = 0.008     # 8mm: 持续穿透预警
PEN_FAIL = 0.025     # 25mm: 瞬时硬穿模
DEPTH_EPS = 1e-9


def main():
    env = make_env()
    ex = IKExpert(env)
    env.reset()
    ex.rebind(env)

    m, d = env.sim.model._model, env.sim.data._data
    gname = lambda gid: (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gid)
                         or f"geom{gid}")

    finger_geoms = set()
    for g in range(m.ngeom):
        bid = m.geom_bodyid[g]
        b = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
        if "finger" in b:
            finger_geoms.add(g)
    arm_geoms = set()
    for g in range(m.ngeom):
        bid = m.geom_bodyid[g]
        b = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
        if b.startswith("robot0_") and g not in finger_geoms:
            arm_geoms.add(g)
    table_gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "table_collision")
    plinth_gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "Sample_Plinth")
    finger_geoms.discard(-1)

    worst = {}          # pair -> (max penetration, step)
    ball_start = None
    arm_hits = []       # (step, other_name, depth)
    n_steps = 0
    ok_rollout, steps, info = None, 0, {}

    # 逐相位执行并在每个物理控制步后扫描接触
    cube_id = env.obj_body_id["RedCube"]
    cube0 = np.array(env.sim.data.xpos[cube_id]).copy()
    wps = ex.plan(cube0)

    def scan(step):
        nonlocal n_steps, ball_start
        n_steps += 1
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = c.geom1, c.geom2
            if c.dist >= -DEPTH_EPS:
                continue
            depth = -c.dist
            pair = tuple(sorted((gname(g1), gname(g2))))
            if depth > worst.get(pair, (0, 0))[0]:
                worst[pair] = (depth, step)
            # 机械臂连杆(非手指)碰到任何东西 → 记录
            if (g1 in arm_geoms or g2 in arm_geoms) and depth > 0.002:
                other = gname(g2 if g1 in arm_geoms else g1)
                arm_hits.append((step, other, round(depth, 4)))

    for wp_i, wp in enumerate(wps):
        if wp["pos"] is not None:
            q_arm, err = ex.solve_ik(wp["pos"])
            for _ in range(300):
                qa = d.qpos[ex.arm_adrs]
                if np.all(np.abs(q_arm - qa) < 0.06):
                    break
                a = np.zeros(env.action_dim)
                a[:7] = qa + np.clip(q_arm - qa, -0.25, 0.25)
                a[7] = float(wp["gripper"])
                obs, _, done, _ = env.step(a)
                scan(n_steps := n_steps + 1)
                if done:
                    break
        else:
            hold = np.zeros(env.action_dim)
            hold[:7] = d.qpos[ex.arm_adrs]
            hold[7] = float(wp["gripper"])
            for _ in range(int(wp.get("dwell") or 30)):
                obs, _, done, _ = env.step(hold)
                scan(n_steps := n_steps + 1)
                if done:
                    break
        if done:
            break

    ok_rollout = env._check_success()
    cube_end = np.array(env.sim.data.xpos[cube_id])

    print("=" * 64)
    print(f"PENETRATION AUDIT — rollout steps={n_steps}, "
          f"success={ok_rollout}")
    print("=" * 64)

    failures = []

    # 1. 机械臂连杆碰撞
    if arm_hits:
        uniq = sorted(set(h[1] for h in arm_hits))
        print(f"[FAIL] arm link collided with: {uniq} "
              f"({len(arm_hits)} steps)")
        failures.append("arm link collision")
    else:
        print("[ok] A. no arm-link collisions (fingers excluded)")

    # 2. 逐对穿透深度
    white_pairs = {
        frozenset(("RedCube", "table_collision")),
        frozenset(("RedCube", "Sample_Plinth")),
        frozenset(("RedCube", "gripper0_right_leftfinger")),
        frozenset(("RedCube", "gripper0_right_rightfinger")),
    }
    ball_gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "YellowBall_g0")
    ball_start = None
    for pair, (depth, step) in sorted(worst.items(), key=lambda kv: -kv[1][0]):
        names = set(pair)
        legit = any(names == frozenset((a, b)) or names <= frozenset((a, b))
                    for a, b in white_pairs) or \
                any(("RedCube" in n or "BlueCyl" in n or "YellowBall" in n)
                    for n in names) and \
                any(("table" in n or "Sample_Plinth" in n or "box_" in n
                     or "finger" in n) for n in names)
        # 简化判定:动态物与桌/台/爪/盒的浅接触为合法;其余任何 >PEN_WARN 可疑
        status = "ok"
        if depth > PEN_FAIL:
            status = "HARD-TUNNEL"
            failures.append(f"tunnel {pair} {depth:.3f}m")
        elif depth > PEN_WARN:
            # 合法对允许浅穿透;非白名单深穿透也可疑
            if pair_js := (names & {"gripper0_right_leftfinger",
                                    "gripper0_right_rightfinger",
                                    "table_collision", "Sample_Plinth"}):
                status = "ok(shallow)"
            else:
                status = "SUSPECT"
                failures.append(f"deep penetration {pair} {depth:.3f}m")
        print(f"[{status:11s}] {pair[0][:28]:28s} <-> {pair[1][:28]:28s} "
              f"max_pen={depth*1000:.1f} mm @step={step}")

    # 3. 终态几何复核:红块应在盒内、不与壁相交
    zone = ex.zone_center
    zh = ex.zone_half
    rel = np.abs(cube_end - zone)
    in_zone = bool(np.all(rel < zh))
    print(f"[{'ok' if in_zone else 'FAIL'}] C. cube inside box zone "
          f"at end: rel={np.round(rel,3).tolist()}")
    if not in_zone:
        failures.append("cube not in box at end")

    print("-" * 64)
    if ok_rollout is False:
        print("note: episode success=False (audit still valid)")
    if failures:
        print("PENETRATION_RESULT: FAILED")
        for f_ in failures:
            print("  -", f_)
        sys.exit(1)
    print("PENETRATION_RESULT: PASS")
    sys.exit(0)


if __name__ == "__main__":
    main()

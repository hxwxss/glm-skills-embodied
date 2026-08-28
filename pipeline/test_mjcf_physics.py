# -*- coding: utf-8 -*-
"""
test_mjcf_physics.py — 编译产物 lab_scene.xml 的无头物理自检
==============================================================

检查项（模拟 3 秒 @ 500Hz）：
  A. 模型加载成功，包含全部期望物体
  B. 所有动态物体稳定落在桌面高度（不穿桌、不弹飞）
  C. 动态物体两两间无穿透（AABB 分离）
  D. 仿真结束时接近静止（settle）
  E. 红块未被初始化进盒内（起点在桌面任务区外）
  F. 离屏渲染一张顶部视图存到 renders/mujoco_settle.png（视觉自检）

用法：
    python test_mjcf_physics.py            # 使用默认路径
退出码 0 = PASS, 1 = FAIL
"""

import os
import sys
import json

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
XML = os.path.join(HERE, "generated", "lab_scene.xml")
SHOT = os.path.join(HERE, "..", "renders", "mujoco_settle.png")

# 黄球已静态化(arena geom),动态体仅红块与蓝柱
DYNAMIC_BODIES = ["Prop_Cube_Red", "Prop_Cylinder_Blue"]
TABLE_TOP_Z = 0.75
STEPS = 1500          # 0.002s * 1500 = 3.0s


def aabb_overlap(p1, s1, p2, s2):
    """两个 AABB(half-extents) 是否重叠。"""
    return all(abs(p1[i] - p2[i]) < (s1[i] + s2[i]) for i in range(3))


def body_aabb(model, data, body_id):
    """body 的世界中心与 half extents（geom world bbox 的近似，放宽 10%）。"""
    positions, sizes = [], []
    for g in range(model.ngeom):
        if model.geom_bodyid[g] == body_id:
            gp = data.geom_xpos[g]
            gm = model.geom_size[g]
            gt = model.geom_type[g]
            if gt == mujoco.mjtGeom.mjGEOM_BOX:
                s = gm[:3].copy()
            elif gt == mujoco.mjtGeom.mjGEOM_CYLINDER:
                s = np.array([gm[0], gm[0], gm[1]])
            elif gt == mujoco.mjtGeom.mjGEOM_SPHERE:
                s = np.array([gm[0]] * 3)
            else:
                continue
            positions.append(gp)
            sizes.append(s * 1.1)
    return np.array(positions), np.array(sizes)


def main():
    failures = []
    print("=" * 62)
    print("PHYSICS SELF-CHECK:", XML)
    print("=" * 62)

    model = mujoco.MjModel.from_xml_path(XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    print("[ok] model loaded; nq=%d ngeom=%d nbody=%d"
          % (model.nq, model.ngeom, model.nbody))

    # A. 物体存在
    bid = {}
    for name in DYNAMIC_BODIES:
        b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        (failures.append(f"missing body {name}") if b < 0 else None)
        bid[name] = b
    print("[ok] dynamic bodies present:", list(bid))
    if failures:
        print("FAIL"); sys.exit(1)

    # 记录初始位置
    x0 = {n: data.body(bid[n]).xpos.copy() for n in DYNAMIC_BODIES}

    # E. 红块起点必须在盒外
    box_c = np.array([0.62, 0.16])
    red_xy = x0["Prop_Cube_Red"][:2]
    outside = np.linalg.norm(red_xy - box_c) > 0.20
    check_e = bool(outside)
    print("[%s] E. red cube starts outside the box (dist=%.3f m)"
          % ("ok" if check_e else "FAIL", np.linalg.norm(red_xy - box_c)))
    if not check_e:
        failures.append("red cube starts inside/near box")

    # B/C/D. 模拟并检查
    final_speed = {}
    for step in range(STEPS):
        mujoco.mj_step(model, data)
        if step % 250 == 0:
            zs = [data.body(bid[n]).xpos[2] for n in DYNAMIC_BODIES]
            print("   t=%.2fs z=%s" % (step * 0.002,
                  ["%.3f" % v for v in zs]))

    # B. 高度检查：红/蓝/黄最终都应位于桌面附近或盒内（只允许 >= 地板且 <= 桌上若干）
    for name in DYNAMIC_BODIES:
        z = float(data.body(bid[name]).xpos[2])
        ok_b = (z > TABLE_TOP_Z - 0.06) and (z < TABLE_TOP_Z + 0.30)
        print("[%s] B. %s settled height z=%.3f" % ("ok" if ok_b else "FAIL", name, z))
        if not ok_b:
            failures.append(f"{name} fell below table or flew")

    # C. 两两穿透（世界 AABB）
    boxes = {}
    for name in DYNAMIC_BODIES:
        pos_arr, size_arr = body_aabb(model, data, bid[name])
        c = data.body(bid[name]).xpos
        ext = size_arr.max(axis=0) if len(size_arr) else np.array([0.03]*3)
        boxes[name] = (np.array(c), np.array(ext))
    pairs = [("Prop_Cube_Red", "Prop_Cylinder_Blue")]
    for a, b in pairs:
        pa, sa = boxes[a]; pb_, sb = boxes[b]
        overlap = all(abs(pa[i]-pb_[i]) < (sa[i]+sb[i]) * 0.98 for i in range(3))
        print("[%s] C. no penetration %s vs %s" % ("FAIL" if overlap else "ok", a, b))
        if overlap:
            failures.append(f"penetration {a}/{b}")

    # D. 结束时速度
    for name in DYNAMIC_BODIES:
        v = float(np.linalg.norm(data.body(bid[name]).cvel[3:6]))
        final_speed[name] = round(v, 4)
        ok_d = v < 0.05
        print("[%s] D. %s speed=%.4f m/s" % ("ok" if ok_d else "FAIL", name, v))
        if not ok_d:
            failures.append(f"{name} still moving at end")

    # F. 离屏渲染一张图
    shot_ok = render_shot(model, data, SHOT)
    print("[%s] F. offscreen render -> %s" % ("ok" if shot_ok else "SKIP",
                                              SHOT if shot_ok else "(renderer unavailable)"))

    print("-" * 62)
    if failures:
        print("RESULT: PHYSICS_CHECK_FAILED")
        for f in failures:
            print("  -", f)
        sys.exit(1)
    print("RESULT: PHYSICS_OK  speeds=%s" % json.dumps(final_speed))
    sys.exit(0)


def render_shot(model, data, out_png):
    try:
        renderer = mujoco.Renderer(model, height=480, width=640)
        cam = mujoco.MjvCamera()
        cam.lookat[:] = [0.25, 0.05, 0.72]
        cam.distance = 2.0
        cam.elevation = -38
        cam.azimuth = -125
        renderer.update_scene(data, camera=cam)
        img = renderer.render()
        os.makedirs(os.path.dirname(out_png), exist_ok=True)
        try:
            from PIL import Image
            Image.fromarray(img).save(out_png)
        except ImportError:
            import cv2
            cv2.imwrite(out_png, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        renderer.close()
        return True
    except Exception as exc:
        print("   (render failed:", exc, ")")
        return False


if __name__ == "__main__":
    main()

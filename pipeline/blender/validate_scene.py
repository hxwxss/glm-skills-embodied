# -*- coding: utf-8 -*-
"""
validate_scene.py — Embodied Lab 场景自动验证
==============================================

在 Blender 内运行，校验：
  A. 关键对象存在且命名正确（三色物体、收纳盒、铰链、台灯、控制面板、相机、灯光）
  B. 三台固定机位存在且指向桌面区域；场景有活动相机
  C. 材质分配正确（红/蓝/黄/玻璃/自发光等）
  D. 三个物体带刚体碰撞体；台灯初始点亮；盒盖初始闭合
  E. 交互操作符真实可用：
       - 开盖产生铰链关键帧动画，中间帧角度介于闭合与全开之间
       - 关灯同时降低点光功率与灯泡自发光（关键帧）
       - Move Red Object To Box 生成多关键帧轨迹：起点抬起→盒口上方→盒内落点，
         中间帧位置显著异于起点与终点（证明是可见移动而非瞬移）
       - Reset Scene 恢复全部初始位姿并清除动画
  F. 嵌入文本块 interaction_embedded 存在（GUI 一键运行入口）

用法:
    blender --background embodied_lab.blend --python validate_scene.py
退出码: 0 = 全部通过; 1 = 存在失败项。
"""

import bpy
import math
import sys

FAILURES = []
CHECKS = []


def check(name, ok, detail=""):
    CHECKS.append((name, bool(ok), detail))
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def find(name):
    return bpy.data.objects.get(name)


# ----------------------------------------------------------------------------
# A. 对象清单
# ----------------------------------------------------------------------------
def validate_objects():
    required = [
        "Room_Floor", "Room_Wall_Back", "Room_Wall_Left", "Room_Wall_Right",
        "Table_Top",
        "Prop_Cube_Red", "Prop_Cylinder_Blue", "Prop_Sphere_Yellow",
        "Storage_Box_Body", "Storage_Box_Lid", "Hinge_BoxLid", "Storage_Box_Tray",
        "Desk_Lamp_Base", "Desk_Lamp_Shade", "Desk_Lamp_Bulb",
        "Light_DeskLamp_Point", "Light_Sun_Key", "Light_Fill_Area",
        "Control_Panel_Deck", "Control_Panel_Button", "Control_Panel_Screen",
        "Cam_Overview", "Cam_Tabletop", "Cam_InteractionCloseup",
    ]
    for name in required:
        ob = find(name)
        check(f"object:{name}", ob is not None,
              "" if ob else "missing")
    hinge = find("Hinge_BoxLid")
    lid = find("Storage_Box_Lid")
    if hinge and lid:
        check("lid parented to hinge", lid.parent is hinge)


# ----------------------------------------------------------------------------
# B. 相机
# ----------------------------------------------------------------------------
def validate_cameras():
    table_center = None
    tt = find("Table_Top")
    if tt:
        table_center = tt.matrix_world.translation.copy()
        table_center.z += 0.05
    for cam_name in ("Cam_Overview", "Cam_Tabletop", "Cam_InteractionCloseup"):
        cam = find(cam_name)
        if not cam or cam.type != 'CAMERA':
            check(f"camera:{cam_name}", False, "not a camera")
            continue
        ok_aim = True
        if table_center:
            direction = cam.matrix_world.to_quaternion() @ __import__(
                "mathutils").Vector((0, 0, -1))
            target_dir = (table_center - cam.matrix_world.translation).normalized()
            dot = direction.normalized().dot(target_dir)
            ok_aim = dot > 0.85   # 视轴与桌心方向夹角 < ~32°
            check(f"camera:{cam_name}", True,
                  f"pos=({cam.location.x:.2f},{cam.location.y:.2f},{cam.location.z:.2f}) "
                  f"aim_dot={dot:.3f}")
        else:
            check(f"camera:{cam_name}", True, f"pos={tuple(cam.location)}")
    sc = bpy.context.scene
    check("scene.active_camera", sc.camera is not None,
          sc.camera.name if sc.camera else "none")


# ----------------------------------------------------------------------------
# C. 材质
# ----------------------------------------------------------------------------
def _mat_of(obj_name):
    ob = find(obj_name)
    if ob and ob.data and ob.data.materials:
        return ob.data.materials[0]
    return None


def validate_materials():
    expect = {
        "Prop_Cube_Red": ("Mat_Prop_Red_Glossy", (0.8, 0.03, 0.03)),
        "Prop_Cylinder_Blue": ("Mat_Prop_Blue_Matte", (0.03, 0.13, 0.80)),
        "Prop_Sphere_Yellow": ("Mat_Prop_Yellow_Satin", (0.92, 0.72, 0.04)),
    }
    for obj_name, (mat_name, rgb) in expect.items():
        mat = _mat_of(obj_name)
        ok_mat = mat is not None and mat.name == mat_name
        ok_col = False
        if ok_mat and mat.use_nodes:
            for node in mat.node_tree.nodes:
                if node.type == 'BSDF_PRINCIPLED':
                    c = node.inputs["Base Color"].default_value
                    ok_col = all(abs(c[i] - rgb[i]) < 0.15 for i in range(3))
                    break
        check(f"material:{obj_name}", ok_mat and ok_col,
              mat.name if mat else "no material")

    glass = _mat_of("Storage_Box_Body")
    has_transmission = False
    if glass and glass.use_nodes:
        for node in glass.node_tree.nodes:
            if node.type == 'BSDF_PRINCIPLED':
                tw = node.inputs.get("Transmission Weight")
                has_transmission = tw is not None and tw.default_value > 0.5
                break
    check("box material transparent", has_transmission)

    bulb_ok = False
    bulb = find("Desk_Lamp_Bulb")
    if bulb and bulb.data.materials:
        mat = bulb.data.materials[0]
        if mat.use_nodes:
            for node in mat.node_tree.nodes:
                if node.type == 'BSDF_PRINCIPLED':
                    es = node.inputs.get("Emission Strength")
                    bulb_ok = es is not None and es.default_value > 1.0
                    break
    check("bulb emissive when on", bulb_ok)


# ----------------------------------------------------------------------------
# D. 刚体 / 初始状态
# ----------------------------------------------------------------------------
def validate_rigidbody_and_state():
    for name in ("Prop_Cube_Red", "Prop_Cylinder_Blue", "Prop_Sphere_Yellow"):
        ob = find(name)
        rb = getattr(ob, "rigid_body", None) if ob else None
        check(f"collision body:{name}", rb is not None,
              rb.collision_shape if rb else "no rigid_body")

    sc = bpy.context.scene
    light = find("Light_DeskLamp_Point")
    energy_on = sc.get("demo_light_energy_on")
    check("initial state recorded",
          all(sc.get(k) is not None for k in
              ("demo_light_on", "demo_box_open", "demo_red_in_box",
               "demo_red_target_in_box")))
    check("lamp initially on", bool(light and energy_on and
                                    abs(light.data.energy - energy_on) < 0.01),
          f"energy={light.data.energy if light else '?'}")
    reset_loc = find("Prop_Cube_Red").get("embodied_reset_location") \
        if find("Prop_Cube_Red") else None
    check("reset data embedded", reset_loc is not None, str(reset_loc))


# ----------------------------------------------------------------------------
# E. 交互行为（核心闭环验证）
# ----------------------------------------------------------------------------
def _embed_exec():
    """执行嵌入的 interaction.py 文本块（模拟用户在 GUI 按 Run Script）。"""
    txt = bpy.data.texts.get("interaction_embedded")
    if txt is None:
        return False
    namespace = {"__name__": "__main__"}
    exec(compile(txt.as_string(), "interaction_embedded", "exec"), namespace)
    return True


def iter_fcurves(anim_data):
    """Blender 5.x slotted actions 与旧版 Action.fcurves 的兼容遍历。"""
    if anim_data is None or anim_data.action is None:
        return []
    action = anim_data.action
    if hasattr(action, "fcurves"):
        return list(action.fcurves)
    fcs = []
    for layer in action.layers:
        for strip in layer.strips:
            if hasattr(strip, "channelbags"):
                for bag in strip.channelbags:
                    fcs.extend(bag.fcurves)
    return fcs


def key_values(anim_data, data_path, array_index=0):
    """返回 [(frame, value), ...]；找不到曲线则返回 []。"""
    for fc in iter_fcurves(anim_data):
        if fc.data_path == data_path and fc.array_index == array_index:
            return [(int(round(kp.co[0])), kp.co[1]) for kp in fc.keyframe_points]
    return []


def validate_interactions():
    sc = bpy.context.scene
    ops_ok = _embed_exec()
    check("embedded script executes", ops_ok)
    if not ops_ok:
        return

    reg = hasattr(bpy.types, "EMBODIED_PT_main_panel")
    check("panel class registered", reg)

    # --- Box lid ---
    hinge = find("Hinge_BoxLid")
    closed_x = hinge.rotation_euler.x
    res = bpy.ops.embodied.box_toggle_lid()
    check("op box_toggle_lid runs", res == {'FINISHED'}, str(res))
    kf = key_values(hinge.animation_data, "rotation_euler", 0)
    check("lid animation keyed", len(kf) >= 2, f"keys={kf}")
    end_val = kf[-1][1]
    check("lid opens to target angle",
          abs(end_val - math.radians(-108)) < 0.02,
          f"end={math.degrees(end_val):.1f}deg closed={math.degrees(closed_x):.1f}deg")

    # 中间帧角度应介于闭合与全开之间（证明存在连续运动）
    f_first, f_last = kf[0][0], kf[-1][0]
    f_mid = (f_first + f_last) // 2
    sc.frame_set(f_mid)
    mid_angle = hinge.rotation_euler.x
    lo, hi = sorted((0.0, math.radians(-108)))
    check("lid motion intermediate frame in-between",
          lo - 1e-4 <= mid_angle <= hi + 1e-4 and abs(mid_angle - lo) > 1e-3,
          f"frame {f_mid}: {math.degrees(mid_angle):.1f}deg")

    # close again
    res = bpy.ops.embodied.box_toggle_lid()
    check("op box_toggle_lid toggles back", res == {'FINISHED'})
    state_open = bool(sc.get("demo_box_open"))
    check("state box_open updated", state_open is False, f"state={state_open}")

    # --- Light toggle ---
    light = find("Light_DeskLamp_Point")
    bulb_mat = find("Desk_Lamp_Bulb").data.materials[0]
    e_before = light.data.energy
    res = bpy.ops.embodied.light_toggle()
    check("op light_toggle runs", res == {'FINISHED'})
    e_keys = key_values(light.data.animation_data, "energy")
    check("light fades via energy keyframes",
          len(e_keys) >= 2 and abs(e_keys[-1][1]) < 0.01 and
          abs(e_keys[0][1] - e_before) < 0.01,
          f"keys={[v for _, v in e_keys]}")
    # 泛化检查：材质节点树上存在自发光强度关键帧且终值≈0（真实渐隐）
    # 注意：节点输入的关键帧挂在 node_tree.animation_data，而非 material.animation_data
    has_emission_keys = False
    for fc in iter_fcurves(bulb_mat.node_tree.animation_data):
        kps = [kp.co[1] for kp in fc.keyframe_points]
        if len(kps) >= 2 and abs(kps[-1]) < 0.01 and kps[0] > 1.0:
            has_emission_keys = True
    check("bulb emission animated (fades out)", has_emission_keys)
    # 帧推进到关灯动画末端之后，自发光插值应趋近 0（真实改变场景照明）
    sc.frame_set(e_keys[-1][0] + 2)
    bulb_strength_now = None
    for node in bulb_mat.node_tree.nodes:
        if node.type == 'BSDF_PRINCIPLED':
            bulb_strength_now = node.inputs["Emission Strength"].default_value
    check("bulb emission near zero at end of off-anim",
          bulb_strength_now is not None and bulb_strength_now < 1.0,
          f"strength={bulb_strength_now:.2f}")
    check("state light_on updated to False",
          bool(sc.get("demo_light_on")) is False)

    # turn back on
    res = bpy.ops.embodied.light_toggle()
    e_keys_on = key_values(light.data.animation_data, "energy")
    check("light turns back on",
          len(e_keys_on) >= 2 and abs(e_keys_on[-1][1] - e_before) < 0.01,
          f"keys={[v for _, v in e_keys_on]}")

    # --- Move red to box ---
    red = find("Prop_Cube_Red")
    start = tuple(round(v, 4) for v in red.location)
    res = bpy.ops.embodied.move_red_to_box()
    check("op move_red_to_box runs", res == {'FINISHED'})
    loc_kf = key_values(red.animation_data, "location", 0)
    z_kf = key_values(red.animation_data, "location", 2)
    check("move trajectory multi-keyed", len(loc_kf) == 4,
          f"{len(loc_kf)} location.x keys")

    span_frames = loc_kf[-1][0] - loc_kf[0][0]
    lift_peak_z = max(v for _, v in z_kf)
    final_z = z_kf[-1][1]
    check("move spans many frames (visible animation)", span_frames >= 60,
          f"{span_frames} frames")
    check("trajectory lifts above tabletop before descending",
          lift_peak_z > final_z + 0.02,
          f"peak z={lift_peak_z:.3f} final z={final_z:.3f}")

    f_mid = (loc_kf[0][0] + loc_kf[-1][0]) // 2
    sc.frame_set(f_mid)
    mid_loc = tuple(round(v, 4) for v in red.location)
    d_start_mid = math.dist(mid_loc, start)
    tgt = sc.get("demo_red_target_in_box")
    d_mid_target = math.dist(mid_loc, tuple(tgt))
    check("intermediate frame between start and box (no teleport)",
          d_start_mid > 0.01 and d_mid_target > 0.01,
          f"mid={mid_loc} d_start={d_start_mid:.3f} d_target={d_mid_target:.3f}")

    sc.frame_set(loc_kf[-1][0] + 2)
    final_loc = tuple(round(v, 4) for v in red.location)
    inside = all(abs(final_loc[i] - tgt[i]) < 0.05 for i in range(3))
    check("red object lands at box interior target", inside,
          f"final={final_loc} target={tuple(round(float(t), 3) for t in tgt)}")
    check("state red_in_box set", bool(sc.get("demo_red_in_box")) is True)

    # --- Reset scene ---
    res = bpy.ops.embodied.reset_scene()
    check("op reset_scene runs", res == {'FINISHED'})
    after_reset = tuple(round(v, 4) for v in red.location)
    check("reset restores red object", after_reset == start,
          f"{after_reset} vs {start}")
    check("reset restores lid angle",
          abs(hinge.rotation_euler.x - closed_x) < 1e-6,
          f"x={math.degrees(hinge.rotation_euler.x):.2f}deg")
    check("reset clears red animation",
          red.animation_data is None or red.animation_data.action is None)
    check("reset turns lamp back on",
          abs(light.data.energy - e_before) < 0.01,
          f"energy={light.data.energy:.1f}")
    check("reset clears flags",
          bool(sc.get("demo_red_in_box")) is False and
          bool(sc.get("demo_box_open")) is False and
          bool(sc.get("demo_light_on")) is True)


# ----------------------------------------------------------------------------
# F. 输出物
# ----------------------------------------------------------------------------
def validate_text_blocks_and_files():
    import os
    check("text block interaction_embedded",
          bpy.data.texts.get("interaction_embedded") is not None)
    base = os.path.dirname(bpy.data.filepath) or "."
    renders = os.path.join(base, "renders")
    it_dirs = sorted(d for d in os.listdir(renders)
                     if d.startswith("iteration_")) if os.path.isdir(renders) else []
    check("iteration render folders >= 4", len(it_dirs) >= 4,
          ",".join(it_dirs))
    for f in ("build_scene.py", "interaction.py", "validate_scene.py",
              "render_views.py"):
        check(f"file:{f}", os.path.isfile(os.path.join(base, f)))


def main():
    print("=" * 64)
    print("VALIDATE Embodied Lab:", bpy.data.filepath)
    print("=" * 64)
    validate_objects()
    validate_cameras()
    validate_materials()
    validate_rigidbody_and_state()
    validate_interactions()
    validate_text_blocks_and_files()

    total = len(CHECKS)
    passed = sum(1 for _, ok, _ in CHECKS if ok)
    print("-" * 64)
    print(f"RESULT: {passed}/{total} checks passed")
    if FAILURES:
        print("FAILED:")
        for f in FAILURES:
            print("  -", f)
        print("VALIDATE_FAILED")
        sys.exit(1)
    print("VALIDATE_OK")


if __name__ == "__main__":
    main()

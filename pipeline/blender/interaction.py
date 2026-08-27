# -*- coding: utf-8 -*-
"""
interaction.py — Embodied Demo 交互模块
=======================================

提供 3D View N 面板 "Embodied Demo" 控制面板，包含四个操作：

  * Open Box / Close Box      盒盖绕铰链开合（带 20 帧动画）
  * Light On / Off            台灯开关（真实改变点光功率 + 灯泡自发光，10 帧渐变）
  * Move Red Object To Box    红色方块抬起→越过盒口→落入盒内的 72 帧可见动画
  * Reset Scene               恢复全部初始状态并清除本模块产生的动画

运行方式（Blender 不保存运行时注册，重新打开 .blend 后需重新执行一次本脚本）：
  1) 打开 embodied_lab.blend；
  2) 切到 Scripting 工作区，文本编辑器中选择内置文本块 "interaction_embedded"
     （或打开本文件），点击 ▶ Run Script（Alt+P）；
  3) 回到 Layout 工作区，按 N 打开侧边栏 → "Embodied Demo" 标签页。
     点击按钮后按空格键播放时间线即可看到动画。

所有对象名与 build_scene.py 中的常量一一对应；初始状态由构建脚本写入
场景自定义属性（demo_*），重置时从这里恢复。
"""

import bpy
import math
import json

# ---- 与 build_scene.py 保持一致的对象名 ----
RED_OBJ = "Prop_Cube_Red"
BLUE_OBJ = "Prop_Cylinder_Blue"
YELLOW_OBJ = "Prop_Sphere_Yellow"
LID_HINGE = "Hinge_BoxLid"
LAMP_LIGHT = "Light_DeskLamp_Point"
LAMP_BULB = "Desk_Lamp_Bulb"
TABLE_OBJ = "Table_Top"

LID_OPEN_ANGLE = math.radians(-108)   # 绕 X 轴开盖角度（铰链在后缘，负值向前上方掀起）
ANIM_LID_FRAMES = 20                  # 开/关盖动画时长（帧）
ANIM_LIGHT_FRAMES = 10                # 灯光渐变时长（帧）
ANIM_MOVE_FRAMES = 72                 # 红块移动动画时长（24fps ≈ 3 秒）
FPS = 24


# ----------------------------------------------------------------------------
# 状态读写（持久化在场景自定义属性中）
# ----------------------------------------------------------------------------
def _sc():
    return bpy.context.scene


def get_state():
    sc = _sc()
    return {
        "light_on": bool(sc.get("demo_light_on", True)),
        "box_open": bool(sc.get("demo_box_open", False)),
        "red_in_box": bool(sc.get("demo_red_in_box", False)),
    }


def set_state(**kwargs):
    sc = _sc()
    for k, v in kwargs.items():
        sc["demo_" + k] = v
    sc["demo_state_json"] = json.dumps(get_state())


def _find(name):
    return bpy.data.objects.get(name)


def iter_fcurves(anim_data):
    """Blender 5.x slotted actions 与旧版 Action.fcurves 的兼容遍历。

    5.x 中关键帧曲线位于 layers → strips → channelbags → fcurves。
    """
    if anim_data is None or anim_data.action is None:
        return []
    action = anim_data.action
    if hasattr(action, "fcurves"):          # <= 4.3 兼容路径
        return list(action.fcurves)
    fcs = []
    for layer in action.layers:
        for strip in layer.strips:
            if hasattr(strip, "channelbags"):
                for bag in strip.channelbags:
                    fcs.extend(bag.fcurves)
    return fcs


def smooth_keys(anim_data, interpolation='BEZIER'):
    """把 anim_data 关键帧全部设为给定插值方式（用于缓入缓出）。"""
    for fc in iter_fcurves(anim_data):
        for kp in fc.keyframe_points:
            kp.interpolation = interpolation
            if interpolation == 'BEZIER':
                kp.handle_left_type = 'AUTO_CLAMPED'
                kp.handle_right_type = 'AUTO_CLAMPED'


def _set_bulb_emission(strength):
    """设置灯泡材质自发光强度。返回 (socket, input_index) 或 None。"""
    bulb = _find(LAMP_BULB)
    if not bulb or not bulb.data.materials:
        return None
    mat = bulb.data.materials[0]
    if not mat.node_tree:
        return None
    for node in mat.node_tree.nodes:
        if node.type == 'BSDF_PRINCIPLED':
            for idx, inp in enumerate(node.inputs):
                if inp.name == "Emission Strength":
                    inp.default_value = strength
                    return inp, idx
    return None


def _get_bulb_emission():
    bulb = _find(LAMP_BULB)
    if not bulb or not bulb.data.materials:
        return 0.0
    mat = bulb.data.materials[0]
    if not mat.node_tree:
        return 0.0
    for node in mat.node_tree.nodes:
        if node.type == 'BSDF_PRINCIPLED':
            inp = node.inputs.get("Emission Strength")
            if inp:
                return inp.default_value
    return 0.0


# ----------------------------------------------------------------------------
# 操作符：Reset Scene
# ----------------------------------------------------------------------------
class EMBODIED_OT_reset_scene(bpy.types.Operator):
    """恢复所有物体到构建时的初始状态，并清除交互产生的动画"""
    bl_idname = "embodied.reset_scene"
    bl_label = "Reset Scene"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        sc = context.scene
        moved = []
        for name in (RED_OBJ, BLUE_OBJ, YELLOW_OBJ):
            ob = _find(name)
            if not ob:
                continue
            if ob.animation_data:
                ob.animation_data_clear()
            loc = ob.get("embodied_reset_location")
            rot = ob.get("embodied_reset_rotation_xyz")
            if loc:
                ob.location = (loc[0], loc[1], loc[2])
            if rot:
                ob.rotation_euler = (rot[0], rot[1], rot[2])
            moved.append(ob.name)

        hinge = _find(LID_HINGE)
        if hinge:
            if hinge.animation_data:
                hinge.animation_data_clear()
            hinge.rotation_euler = (0.0, 0.0, 0.0)
            moved.append(hinge.name)

        # 台灯恢复初始常亮
        light = _find(LAMP_LIGHT)
        energy_on = sc.get("demo_light_energy_on", 28.0)
        if light and light.data:
            if light.data.animation_data:
                light.data.animation_data_clear()
            light.data.energy = energy_on
        strength_on = sc.get("demo_bulb_emission_on", 42.0)
        _set_bulb_emission(strength_on)
        bulb = _find(LAMP_BULB)
        if bulb and bulb.data.materials:
            mat = bulb.data.materials[0]
            if mat.node_tree and mat.node_tree.animation_data:
                mat.node_tree.animation_data_clear()

        set_state(light_on=True, box_open=False, red_in_box=False)
        sc.frame_set(1)
        self.report({'INFO'}, f"Reset done: {', '.join(moved)}")
        return {'FINISHED'}


# ----------------------------------------------------------------------------
# 操作符：Open / Close Box
# ----------------------------------------------------------------------------
class EMBODIED_OT_box_toggle_lid(bpy.types.Operator):
    """开合收纳盒盖（20 帧动画，按空格播放查看）"""
    bl_idname = "embodied.box_toggle_lid"
    bl_label = "Toggle Box Lid"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return _find(LID_HINGE) is not None

    def execute(self, context):
        sc = context.scene
        hinge = _find(LID_HINGE)
        opening = not get_state()["box_open"]
        target = LID_OPEN_ANGLE if opening else 0.0

        # 清掉旧动画，从当前角度平滑过渡到目标角度
        if hinge.animation_data:
            hinge.animation_data_clear()
        current = hinge.rotation_euler.x
        f0 = sc.frame_current
        f1 = f0 + ANIM_LID_FRAMES
        hinge.rotation_euler.x = current
        hinge.keyframe_insert(data_path="rotation_euler", index=0, frame=f0)
        hinge.rotation_euler.x = target
        hinge.keyframe_insert(data_path="rotation_euler", index=0, frame=f1)
        # 平滑缓入缓出
        smooth_keys(hinge.animation_data)

        sc.frame_end = max(sc.frame_end, f1)
        sc.frame_set(f0)
        set_state(box_open=opening)
        self.report({'INFO'},
                    f"Lid {'opening' if opening else 'closing'}: frames {f0}-{f1}. "
                    f"Press Play/Spacebar to watch.")
        return {'FINISHED'}


# ----------------------------------------------------------------------------
# 操作符：Light On / Off
# ----------------------------------------------------------------------------
class EMBODIED_OT_light_toggle(bpy.types.Operator):
    """开关台灯：同时改变点光功率与灯泡自发光（10 帧渐变）"""
    bl_idname = "embodied.light_toggle"
    bl_label = "Toggle Desk Lamp"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return _find(LAMP_LIGHT) is not None

    def execute(self, context):
        sc = context.scene
        state = get_state()
        turning_on = not state["light_on"]
        light = _find(LAMP_LIGHT)
        energy_on = sc.get("demo_light_energy_on", 28.0)
        strength_on = sc.get("demo_bulb_emission_on", 42.0)
        e_target = energy_on if turning_on else 0.0
        s_target = strength_on if turning_on else 0.0

        f0 = sc.frame_current
        f1 = f0 + ANIM_LIGHT_FRAMES

        if light and light.data:
            light.data.animation_data_clear()
            current = light.data.energy
            light.data.energy = current
            light.data.keyframe_insert(data_path="energy", frame=f0)
            light.data.energy = e_target
            light.data.keyframe_insert(data_path="energy", frame=f1)
            smooth_keys(light.data.animation_data, interpolation='LINEAR')

        bulb = _find(LAMP_BULB)
        if bulb and bulb.data.materials:
            mat = bulb.data.materials[0]
            if mat.node_tree:
                if mat.node_tree.animation_data:
                    mat.node_tree.animation_data_clear()
                result = _set_bulb_emission(_get_bulb_emission())
                if result is not None:
                    inp, idx = result
                    node_name = inp.node.name
                    s_current = inp.default_value
                    path = (f'nodes["{node_name}"]'
                            f'.inputs[{idx}].default_value')
                    inp.default_value = s_current
                    mat.node_tree.keyframe_insert(data_path=path, frame=f0)
                    inp.default_value = s_target
                    mat.node_tree.keyframe_insert(data_path=path, frame=f1)
                    smooth_keys(mat.node_tree.animation_data,
                                interpolation='LINEAR')

        sc.frame_end = max(sc.frame_end, f1)
        sc.frame_set(f0)
        set_state(light_on=turning_on)
        self.report({'INFO'},
                    f"Lamp {'ON' if turning_on else 'OFF'} (frames {f0}-{f1}).")
        return {'FINISHED'}


# ----------------------------------------------------------------------------
# 操作符：Move Red Object To Box（可见动画）
# ----------------------------------------------------------------------------
class EMBODIED_OT_move_red_to_box(bpy.types.Operator):
    """红色方块：抬起→越过盒口→落入盒内（72 帧动画，绝不瞬移）"""
    bl_idname = "embodied.move_red_to_box"
    bl_label = "Move Red Object To Box"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        if _find(RED_OBJ) is None:
            cls.poll_message_set("Prop_Cube_Red not found in scene")
            return False
        if get_state()["red_in_box"]:
            cls.poll_message_set("Red object is already in the box — use Reset Scene")
            return False
        return True

    def execute(self, context):
        sc = context.scene
        red = _find(RED_OBJ)
        if red.animation_data:
            red.animation_data_clear()

        target_list = sc.get("demo_red_target_in_box")
        if not target_list:
            self.report({'ERROR'}, "demo_red_target_in_box missing — rebuild the scene")
            return {'CANCELLED'}
        target = (target_list[0], target_list[1], target_list[2])

        start_loc = tuple(red.location)
        start_yaw = red.rotation_euler.z
        lift = (start_loc[0], start_loc[1], start_loc[2] + 0.10)
        over = (target[0], target[1], target[2] + 0.09)
        end_yaw = start_yaw + math.radians(35)

        f0 = sc.frame_current
        keys = [
            (f0, start_loc, start_yaw),
            (f0 + int(ANIM_MOVE_FRAMES * 0.25), lift, start_yaw + math.radians(8)),
            (f0 + int(ANIM_MOVE_FRAMES * 0.55), over, end_yaw),
            (f0 + ANIM_MOVE_FRAMES, target, end_yaw),
        ]
        for frame, loc, yaw in keys:
            red.location = loc
            red.rotation_euler.z = yaw
            red.keyframe_insert(data_path="location", frame=frame)
            red.keyframe_insert(data_path="rotation_euler", index=2, frame=frame)

        smooth_keys(red.animation_data)

        # 被动碰撞体跟随关键帧（构建时已设 kinematic=True，这里确保一致）
        if red.rigid_body:
            red.rigid_body.kinematic = True

        sc.frame_end = max(sc.frame_end, f0 + ANIM_MOVE_FRAMES)
        sc.frame_set(f0)
        set_state(red_in_box=True)
        self.report({'INFO'},
                    f"Red object moving to box: frames {f0}-{f0 + ANIM_MOVE_FRAMES}. "
                    f"Press Play/Spacebar to watch.")
        return {'FINISHED'}


# ----------------------------------------------------------------------------
# N 面板
# ----------------------------------------------------------------------------
class EMBODIED_PT_main_panel(bpy.types.Panel):
    bl_label = "Embodied Demo"
    bl_idname = "EMBODIED_PT_main_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Embodied Demo"

    def draw(self, context):
        layout = self.layout
        state = get_state()

        box = layout.box()
        box.label(text="Status", icon='INFO')
        box.label(text=f"Lamp: {'ON' if state['light_on'] else 'OFF'}",
                  icon='LIGHT' if state['light_on'] else 'LIGHT_SUN')
        box.label(text=f"Box lid: {'OPEN' if state['box_open'] else 'CLOSED'}",
                  icon='MOD_SOLIDIFY')
        box.label(text=f"Red object: {'in box' if state['red_in_box'] else 'on table'}",
                  icon='MESH_CUBE')

        col = layout.column(align=True)
        col.label(text="Interactions:")
        col.operator(EMBODIED_OT_box_toggle_lid.bl_idname,
                     text="Open Box" if not state["box_open"] else "Close Box",
                     icon='WINDOW' if not state["box_open"] else 'X')
        col.operator(EMBODIED_OT_light_toggle.bl_idname,
                     text="Light On" if not state["light_on"] else "Light Off",
                     icon='OUTLINER_OB_LIGHT' if not state["light_on"] else 'RESTRICT_COLOR_ON')
        props = col.operator(EMBODIED_OT_move_red_to_box.bl_idname,
                             text="Move Red Object To Box", icon='ARMATURE_DATA')
        layout.separator()
        row = layout.row()
        row.operator(EMBODIED_OT_reset_scene.bl_idname, text="Reset Scene", icon='LOOP_BACK')
        layout.separator()
        box2 = layout.box()
        box2.scale_y = 0.7
        box2.label(text="After clicking a button,", icon='PLAY')
        box2.label(text="press Spacebar to play")
        box2.label(text=f"the animation.  FPS: {FPS}")


# ----------------------------------------------------------------------------
# 注册
# ----------------------------------------------------------------------------
classes = (
    EMBODIED_OT_reset_scene,
    EMBODIED_OT_box_toggle_lid,
    EMBODIED_OT_light_toggle,
    EMBODIED_OT_move_red_to_box,
    EMBODIED_PT_main_panel,
)


def register():
    # 支持重复执行脚本（先注销旧类，避免重复注册报错）
    for cls in reversed(classes):
        try:
            bpy.utils.unregister_class(cls)
        except Exception:
            pass
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        try:
            bpy.utils.unregister_class(cls)
        except Exception:
            pass


if __name__ == "__main__":
    register()
    print("[interaction] Embodied Demo panel registered — open N-panel in 3D View.")

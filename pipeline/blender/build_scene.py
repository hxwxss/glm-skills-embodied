# -*- coding: utf-8 -*-
"""
build_scene.py — 小型未来实验室桌面场景（Embodied Lab）构建脚本
用法（无头构建，生成 embodied_lab.blend）:
    blender --background --factory-startup --python build_scene.py

从零重建整个场景：房间、实验桌、三个可交互物体、透明收纳盒（带铰链盖）、
桌面台灯、控制面板、三台固定机位相机、灯光与渲染设置。
所有对象使用语义化稳定命名，初始状态存为自定义属性供 interaction.py / validate_scene.py 使用。

Blender 5.2 / Cycles (OptiX GPU fallback CPU)。
"""

import bpy
import math
import os
import sys
import json
from mathutils import Vector, Euler

# ----------------------------------------------------------------------------
# 常量与布局参数（单位：米）
# ----------------------------------------------------------------------------
FPS = 24
TABLE_TOP_Z = 0.75          # 桌面高度
TABLE_SIZE = (1.90, 0.95)   # 桌面长宽 (x, y)
TABLE_THICK = 0.045

BOX_CENTER = Vector((0.40, 0.12, TABLE_TOP_Z))
BOX_OUTER = (0.27, 0.19, 0.145)   # 收纳盒外尺寸 x,y,z（z 为壁高，不含盖）
BOX_WALL = 0.012
LID_OPEN_ANGLE = math.radians(-108)  # 绕 X 轴开盖角度（铰链在 +Y 后缘，负值向前上方掀起）

PROP_CUBE_RED = "Prop_Cube_Red"
PROP_CYL_BLUE = "Prop_Cylinder_Blue"
PROP_SPH_YELLOW = "Prop_Sphere_Yellow"

# 红块边长受 PandaGripper 实测最大开口(~5.9cm)约束，>5.9cm 无法抓取
CUBE_SIZE = 0.05
CYL_RADIUS = 0.05
CYL_HEIGHT = 0.15
SPH_RADIUS = 0.055

# 红块放在 3cm 样品支撑台上:既抬高抓取高度(指尖包夹上半段),
# 又是实验台的样品架元素。支撑台顶面 = 0.75 + 0.03 = 0.78
PLINTH_H = 0.03
RED_LOC = Vector((0.08, 0.06, TABLE_TOP_Z + PLINTH_H + CUBE_SIZE / 2))
BLUE_LOC = Vector((-0.04, 0.18, TABLE_TOP_Z + 0.075))   # 圆柱 r=5cm h=15cm
YELLOW_LOC = Vector((-0.12, 0.26, TABLE_TOP_Z + 0.055))  # 球 d=11cm;远离爪臂扫掠路径

# 红块放入盒内的落点（盒内部地面之上）
BOX_INNER_FLOOR_Z = BOX_CENTER.z + BOX_WALL
RED_IN_BOX_LOC = Vector((BOX_CENTER.x - 0.01,
                         BOX_CENTER.y + 0.005,
                         BOX_INNER_FLOOR_Z + CUBE_SIZE / 2))

LIGHT_ON_ENERGY = 28.0      # 台灯点光功率 W
BULB_EMISSION_ON = 42.0     # 灯泡自发光强度
CAM_SAMPLES_DEFAULT = 64


def vec(seq):
    return Vector((seq[0], seq[1], seq[2]))


# ----------------------------------------------------------------------------
# 清场与基础工具
# ----------------------------------------------------------------------------
def wipe_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    for coll in (bpy.data.meshes, bpy.data.materials, bpy.data.lights,
                 bpy.data.cameras, bpy.data.objects, bpy.data.worlds):
        for block in list(coll):
            try:
                coll.remove(block, do_unlink=True)
            except Exception:
                pass


def link(obj, collection=None):
    (collection or bpy.context.scene.collection).objects.link(obj)


def new_obj(name, mesh_data):
    obj = bpy.data.objects.new(name, mesh_data)
    link(obj)
    return obj


def box_mesh(name, size_x, size_y, size_z, center=(0, 0, 0)):
    """创建轴对齐长方体 mesh（原点在世界坐标=center）。"""
    import bmesh
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    obj = new_obj(name, me)
    obj.scale = (size_x, size_y, size_z)
    obj.location = vec(center)
    apply_transforms(obj)
    return obj


def cyl_mesh(name, radius, depth, center=(0, 0, 0), rot=(0, 0, 0), verts=48):
    import bmesh
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, segments=verts,
                          radius1=radius, radius2=radius, depth=depth)
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    obj = new_obj(name, me)
    obj.rotation_euler = Euler(rot, 'XYZ')
    obj.location = vec(center)
    return obj


def sph_mesh(name, radius, center=(0, 0, 0), segments=32, rings=16):
    import bmesh
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=segments, v_segments=rings, radius=radius)
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    obj = new_obj(name, me)
    obj.location = vec(center)
    return obj


def apply_transforms(obj, loc=False):
    """把 scale/rotation 应用进 mesh 数据（保持世界变换不变）。"""
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    if loc:
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    else:
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)


def join_objects(objs, name):
    """把多个对象合并为一个。返回合并后的对象（原点在世界原点，网格为世界坐标）。

    关键：join 前必须把所有部件的完整变换（含位移）烘焙进网格。
    否则合并结果沿用 active 对象的原点偏移，后续对 location 赋值会把
    整体几何错位（曾导致盒盖悬空、LED 灯带偏移）。
    """
    objs = [o for o in objs if o and o.name in bpy.data.objects]
    if not objs:
        raise RuntimeError("join_objects: no objects")
    first = objs[0]
    bpy.ops.object.select_all(action='DESELECT')
    for o in objs:
        o.select_set(True)
        bpy.context.view_layer.objects.active = o
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    bpy.ops.object.select_all(action='DESELECT')
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = first
    bpy.ops.object.join()
    first.name = name
    return first


def beam_between(a, b, name, radius, mat):
    """连接两点的圆柱（用于台灯臂）。中点放置，Z 轴沿 a→b。"""
    direction = vec(b) - vec(a)
    length = direction.length
    import bmesh
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, segments=32,
                          radius1=radius, radius2=radius, depth=length)
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    obj = new_obj(name, me)
    quat = direction.normalized().to_track_quat('Z', 'Y')
    obj.rotation_euler = quat.to_euler()
    obj.location = (vec(a) + vec(b)) / 2
    obj.data.materials.append(mat)
    return obj


# ----------------------------------------------------------------------------
# 材质
# ----------------------------------------------------------------------------
def make_material(name, base_color, roughness=0.4, metallic=0.0,
                  emission_color=None, emission_strength=0.0,
                  transmission=0.0, alpha=1.0, ior=1.45):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = None
    out = None
    nt = mat.node_tree
    # 建一棵干净的树：Principled → Output
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(bsdf.outputs[0], out.inputs[0])

    def set_in(socket_name, value):
        s = bsdf.inputs.get(socket_name)
        if s is None:
            for alt in bsdf.inputs:
                pass
            print(f"    [warn] principled input '{socket_name}' missing on {mat.name}")
            return
        try:
            s.default_value = value
        except Exception as e:
            print(f"    [warn] cannot set {socket_name}: {e}")

    c = base_color
    set_in("Base Color", (c[0], c[1], c[2], 1.0))
    set_in("Roughness", roughness)
    set_in("Metallic", metallic)
    set_in("IOR", ior)
    set_in("Transmission Weight", transmission)
    set_in("Alpha", alpha)
    if emission_strength > 0 and emission_color is not None:
        ec = emission_color
        set_in("Emission Color", (ec[0], ec[1], ec[2], 1.0))
        set_in("Emission Strength", emission_strength)

    if alpha < 1.0:
        mat.blend_method = 'BLEND'
    mat.use_backface_culling = False
    return mat


MATS = {}


def build_materials():
    MATS.clear()
    MATS["Floor"] = make_material("Mat_Floor_Concrete", (0.28, 0.285, 0.30), roughness=0.38)
    MATS["Wall"] = make_material("Mat_Wall_Panel", (0.36, 0.37, 0.40), roughness=0.65)
    MATS["WallLightBar"] = make_material("Mat_Wall_LightBar_Emissive", (0.9, 0.95, 1.0),
                                         roughness=0.3, emission_color=(0.75, 0.88, 1.0),
                                         emission_strength=6.0)
    MATS["TableMetal"] = make_material("Mat_Table_Frame_Metal", (0.09, 0.095, 0.105),
                                       roughness=0.35, metallic=0.85)
    MATS["TableTop"] = make_material("Mat_Table_Top_Brush", (0.20, 0.21, 0.23),
                                     roughness=0.33, metallic=0.75)
    MATS["Red"] = make_material("Mat_Prop_Red_Glossy", (0.82, 0.035, 0.03), roughness=0.3)
    MATS["Blue"] = make_material("Mat_Prop_Blue_Matte", (0.035, 0.13, 0.80), roughness=0.45)
    MATS["Yellow"] = make_material("Mat_Prop_Yellow_Satin", (0.92, 0.72, 0.04), roughness=0.22)
    MATS["BoxGlass"] = make_material("Mat_Box_Plastic_Tint", (0.86, 0.95, 0.97),
                                     roughness=0.03, transmission=0.97, ior=1.45)
    MATS["LidFrame"] = make_material("Mat_Box_Lid_Frame", (0.14, 0.15, 0.17),
                                     roughness=0.4, metallic=0.4)
    MATS["RimLED"] = make_material("Mat_Box_Rim_LED", (0.1, 0.9, 0.9), roughness=0.3,
                                   emission_color=(0.15, 0.85, 0.95), emission_strength=2.2)
    MATS["LampBody"] = make_material("Mat_Lamp_OffWhite", (0.55, 0.56, 0.58), roughness=0.4, metallic=0.6)
    MATS["Bulb"] = make_material("Mat_Lamp_Bulb_Emissive", (1.0, 0.98, 0.94), roughness=0.3,
                                 emission_color=(1.0, 0.86, 0.68),
                                 emission_strength=BULB_EMISSION_ON)
    MATS["Panel"] = make_material("Mat_Control_Panel_Deck", (0.10, 0.11, 0.125), roughness=0.5)
    MATS["Screen"] = make_material("Mat_Control_Screen_Emissive", (0.02, 0.05, 0.06), roughness=0.25,
                                   emission_color=(0.10, 0.85, 0.75), emission_strength=2.5)
    MATS["ButtonGreen"] = make_material("Mat_Control_Button_Green", (0.02, 0.25, 0.08), roughness=0.3,
                                        emission_color=(0.05, 0.9, 0.25), emission_strength=3.0)


def set_mat(obj, key):
    if obj and obj.data is not None:
        obj.data.materials.append(MATS[key])


def bevel(obj, width=0.0025, segments=3, angle=50):
    mod = obj.modifiers.new("Bevel", 'BEVEL')
    mod.width = width
    mod.segments = segments
    mod.limit_method = 'ANGLE'
    mod.angle_limit = math.radians(angle)
    mod.harden_normals = False
    return mod


# ----------------------------------------------------------------------------
# 场景元素
# ----------------------------------------------------------------------------
def build_room():
    floor = box_mesh("Room_Floor", 9.0, 9.0, 0.04, (0, 0, -0.02))
    set_mat(floor, "Floor")
    # 后墙加宽到 9m：overview 相机右侧视野曾拍到墙外虚空（死黑区域）
    back = box_mesh("Room_Wall_Back", 9.0, 0.08, 2.7, (0, 2.05, 1.35))
    set_mat(back, "Wall")
    left = box_mesh("Room_Wall_Left", 0.08, 9.0, 2.7, (-2.05, 0, 1.35))
    set_mat(left, "Wall")
    # 右墙段：封住 overview 右侧视野（此前是墙外纯黑虚空）
    right = box_mesh("Room_Wall_Right", 0.08, 4.4, 2.7, (2.05, 0.15, 1.35))
    set_mat(right, "Wall")
    # 后墙中部腰线灯带（未来感装饰 + 环境补光）
    bar = box_mesh("Room_Wall_LightBar", 4.6, 0.05, 0.045, (0, 2.0, 1.85))
    set_mat(bar, "WallLightBar")


def build_table():
    top_z_center = TABLE_TOP_Z - TABLE_THICK / 2
    top = box_mesh("Table_Top", TABLE_SIZE[0], TABLE_SIZE[1], TABLE_THICK,
                   (0, 0, top_z_center))
    set_mat(top, "TableTop")
    bevel(top, width=0.006, segments=4)
    lx, ly = TABLE_SIZE[0] / 2 - 0.09, TABLE_SIZE[1] / 2 - 0.09
    legs = []
    leg_len = TABLE_TOP_Z - TABLE_THICK
    positions = {"FL": (lx, -ly), "FR": (-lx, -ly), "BL": (lx, ly), "BR": (-lx, ly)}
    for tag, (px, py) in positions.items():
        leg = cyl_mesh(f"Table_Leg_{tag}", 0.021, leg_len,
                       (px, py, leg_len / 2), verts=24)
        set_mat(leg, "TableMetal")
        legs.append(leg)
    rail = box_mesh("Table_Rail_Frame", TABLE_SIZE[0] - 0.10, TABLE_SIZE[1] - 0.10, 0.03,
                    (0, 0, TABLE_TOP_Z - TABLE_THICK - 0.02))
    set_mat(rail, "TableMetal")
    return [top] + legs


def smooth_shade(obj):
    """曲面物体平滑着色，消除多边形棱纹。"""
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.shade_smooth()


def build_props():
    cube = box_mesh(PROP_CUBE_RED, CUBE_SIZE, CUBE_SIZE, CUBE_SIZE,
                    (RED_LOC.x, RED_LOC.y, RED_LOC.z))
    set_mat(cube, "Red")
    bevel(cube, width=0.006, segments=3)
    cube.rotation_euler = Euler((0, 0, math.radians(12)), 'XYZ')

    cyl = cyl_mesh(PROP_CYL_BLUE, CYL_RADIUS, CYL_HEIGHT,
                   (BLUE_LOC.x, BLUE_LOC.y, BLUE_LOC.z), verts=64)
    set_mat(cyl, "Blue")
    bevel(cyl, width=0.004, segments=3)
    smooth_shade(cyl)

    sph = sph_mesh(PROP_SPH_YELLOW, SPH_RADIUS,
                   (YELLOW_LOC.x, YELLOW_LOC.y, YELLOW_LOC.z), segments=48, rings=24)
    set_mat(sph, "Yellow")
    smooth_shade(sph)
    for o in (cube, cyl, sph):
        o.rotation_mode = 'XYZ'
    return cube, cyl, sph


def build_storage_box():
    ox, oy, oz = BOX_OUTER
    t = BOX_WALL
    parts = []
    # 底板
    bottom = box_mesh("BoxPart_Bottom", ox, oy, t, (0, 0, t / 2))
    set_mat(bottom, "BoxGlass")
    parts.append(bottom)
    # 前后左右四壁（局部坐标以盒中心为基准）
    walls = [
        ("Front", ox, t, oz, (0, -(oy - t) / 2 - t / 2, t + (oz) / 2)),
        ("Back", ox, t, oz, (0, (oy - t) / 2 + t / 2, t + (oz) / 2)),
        ("Left", t, oy - 2 * t, oz, (-(ox - t) / 2 - t / 2, 0, t + (oz) / 2)),
        ("Right", t, oy - 2 * t, oz, ((ox - t) / 2 + t / 2, 0, t + (oz) / 2)),
    ]
    for tag, sx, sy, sz, ctr in walls:
        w = box_mesh(f"BoxPart_Wall_{tag}", sx, sy, sz, ctr)
        set_mat(w, "BoxGlass")
        parts.append(w)
    body = join_objects(parts, "Storage_Box_Body")
    body.location = vec(BOX_CENTER)
    apply_transforms(body)
    # 不透明底座托盘，强化"收纳盒"轮廓
    tray = box_mesh("Storage_Box_Tray", ox + 0.014, oy + 0.014, 0.010,
                    (BOX_CENTER.x, BOX_CENTER.y, TABLE_TOP_Z + 0.005))
    set_mat(tray, "LidFrame")

    # 内缘 LED 灯带：保证盒内可见、增加未来感（相对盒体局部坐标）
    led_parts = []
    led_h = oz - 0.012
    led = box_mesh("BoxPart_LED_Front", ox - 2 * t - 0.004, 0.006, 0.005,
                   (0, -(oy - t) / 2 - t / 2 + 0.004, t + led_h))
    set_mat(led, "RimLED")
    led_parts.append(led)
    led_b = box_mesh("BoxPart_LED_Back", ox - 2 * t - 0.004, 0.006, 0.005,
                     (0, (oy - t) / 2 + t / 2 - 0.004, t + led_h))
    set_mat(led_b, "RimLED")
    led_parts.append(led_b)
    led_l = box_mesh("BoxPart_LED_Left", 0.006, oy - 2 * t - 0.004, 0.005,
                     (-(ox - t) / 2 - t / 2 + 0.004, 0, t + led_h))
    set_mat(led_l, "RimLED")
    led_parts.append(led_l)
    led_r = box_mesh("BoxPart_LED_Right", 0.006, oy - 2 * t - 0.004, 0.005,
                     ((ox - t) / 2 + t / 2 - 0.004, 0, t + led_h))
    set_mat(led_r, "RimLED")
    led_parts.append(led_r)
    led_frame = join_objects(led_parts, "Storage_Box_RimLED")
    led_frame.location = vec(BOX_CENTER)
    # 注意：不挂父级。盒体自身的 location 位移未烘焙进网格，挂父级会产生双重偏移
    # （LED 曾因此飞到 2×BOX_CENTER）。LED 为静态装饰，独立对象即可。

    # 铰链空物体：位于后缘顶端
    hinge = bpy.data.objects.new("Hinge_BoxLid", None)
    link(hinge)
    hinge.empty_display_size = 0.03
    hinge.location = (BOX_CENTER.x, BOX_CENTER.y + oy / 2, BOX_CENTER.z + oz)
    hinge.rotation_euler = Euler((0, 0, 0), 'XYZ')

    # 盖子：深色外框 4 条 + 内嵌玻璃板 + 前缘把手（开合时轮廓清晰可读）
    fx, fy = ox + 0.010, oy + 0.010      # 外框尺寸
    fw = 0.014                            # 框条宽
    ft = 0.012                            # 框条厚
    lid_parts = []
    fb = box_mesh("LidPart_Frame_Back", fx, fw, ft, (0, fy / 2 - fw / 2, 0))
    set_mat(fb, "LidFrame")
    lid_parts.append(fb)
    ff = box_mesh("LidPart_Frame_Front", fx, fw, ft, (0, -fy / 2 + fw / 2, 0))
    set_mat(ff, "LidFrame")
    lid_parts.append(ff)
    fl = box_mesh("LidPart_Frame_Left", fw, fy - 2 * fw, ft, (-fx / 2 + fw / 2, 0, 0))
    set_mat(fl, "LidFrame")
    lid_parts.append(fl)
    fr = box_mesh("LidPart_Frame_Right", fw, fy - 2 * fw, ft, (fx / 2 - fw / 2, 0, 0))
    set_mat(fr, "LidFrame")
    lid_parts.append(fr)
    board = box_mesh("LidPart_Glass", fx - 2 * fw, fy - 2 * fw, 0.005, (0, 0, 0))
    set_mat(board, "BoxGlass")
    lid_parts.append(board)
    knob = box_mesh("LidPart_Knob", 0.06, 0.016, 0.016, (0, -fy / 2 + 0.03, ft / 2 + 0.006))
    set_mat(knob, "LidFrame")
    lid_parts.append(knob)
    lid = join_objects(lid_parts, "Storage_Box_Lid")
    lid.parent = hinge
    # 盖子网格已在世界坐标（板中心=盒中心原点）。相对铰链的偏移 = 盒中心到
    # 铰链（后缘顶）的向量取反：y=-oy/2，抬高 7.5mm 避免与盒壁穿插。
    lid.location = Vector((0, -oy / 2, 0.0075))
    apply_transforms(lid)
    lid.matrix_parent_inverse.identity()
    bevel(body, width=0.002, segments=2)
    bevel(lid, width=0.002, segments=2)
    return hinge, lid, body


def build_desk_lamp():
    base_c = Vector((-0.75, 0.10, TABLE_TOP_Z))
    base = cyl_mesh("Desk_Lamp_Base", 0.085, 0.024, (base_c.x, base_c.y, base_c.z + 0.012))
    set_mat(base, "LampBody")
    joint_a = Vector((base_c.x, base_c.y, base_c.z + 0.032))

    shoulder_dir = Vector((0.32, -0.18, 0.52)).normalized()
    arm1_end = joint_a + shoulder_dir * 0.36
    arm1 = beam_between(joint_a, arm1_end, "Desk_Lamp_Arm_Lower", 0.011, MATS["LampBody"])

    elbow = arm1_end
    head_target_dir = Vector((0.62, 0.12, -0.34)).normalized()
    head_pos = elbow + head_target_dir * 0.34
    arm2 = beam_between(elbow, head_pos, "Desk_Lamp_Arm_Upper", 0.010, MATS["LampBody"])

    # 灯罩：锥形碗，开口朝向桌面上物体区
    shade_rot_q = (-head_target_dir).to_track_quat('Z', 'Y')  # Z 轴指向开口方向(反向=朝下前方)
    shade_rot = shade_rot_q.to_euler()
    import bmesh
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=False, segments=48,
                          radius1=0.075, radius2=0.022, depth=0.115)
    me = bpy.data.meshes.new("Desk_Lamp_Shade")
    bm.to_mesh(me)
    bm.free()
    shade = new_obj("Desk_Lamp_Shade", me)
    shade.rotation_euler = shade_rot
    shade.location = head_pos - head_target_dir * 0.02
    set_mat(shade, "LampBody")
    bpy.ops.object.shade_smooth()

    bulb = sph_mesh("Desk_Lamp_Bulb", 0.024,
                    tuple(shade.location - head_target_dir * 0.038))
    set_mat(bulb, "Bulb")

    light_data = bpy.data.lights.new("Light_DeskLamp_PointData", 'POINT')
    light_data.energy = LIGHT_ON_ENERGY
    light_data.color = (1.0, 0.83, 0.66)
    light_data.shadow_soft_size = 0.035
    light = bpy.data.objects.new("Light_DeskLamp_Point", light_data)
    link(light)
    light.location = tuple(shade.location - head_target_dir * 0.075)

    solidify = shade.modifiers.new("Solidify", 'SOLIDIFY')
    solidify.thickness = 0.006
    solidify.offset = -1.0
    return {"bulb": bulb, "light": light}


def build_control_panel():
    pc = Vector((-0.38, -0.05, TABLE_TOP_Z))
    deck = box_mesh("Control_Panel_Deck", 0.20, 0.13, 0.022,
                    (pc.x, pc.y, pc.z + 0.075))
    deck.rotation_euler = Euler((math.radians(-10), 0, math.radians(8)), 'XYZ')
    set_mat(deck, "Panel")
    bevel(deck, width=0.004, segments=3)

    foot1 = box_mesh("Control_Panel_Foot", 0.06, 0.10, 0.064, (pc.x - 0.05, pc.y, pc.z + 0.032))
    foot2 = box_mesh("Control_Panel_Foot_Rear", 0.06, 0.10, 0.064, (pc.x + 0.05, pc.y, pc.z + 0.032))
    for f in (foot1, foot2):
        f.rotation_euler = deck.rotation_euler.copy()
        set_mat(f, "LampBody")

    # 屏幕：贴在面板上表面前部（用父级简化定位）
    screen = box_mesh("Control_Panel_Screen", 0.10, 0.05, 0.003,
                      (pc.x - 0.03, pc.y - 0.02, pc.z + 0.0905))
    screen.rotation_euler = deck.rotation_euler.copy()
    set_mat(screen, "Screen")

    button = cyl_mesh("Control_Panel_Button", 0.017, 0.014,
                      (pc.x + 0.06, pc.y + 0.01, pc.z + 0.094), rot=(math.radians(-10), 0, 0))
    set_mat(button, "ButtonGreen")
    return deck


def build_lights_world():
    # 主太阳光：从房间开放角（南-西上方）射入，直接照亮桌面
    sun_data = bpy.data.lights.new("Light_Sun_KeyData", 'SUN')
    sun_data.energy = 5.5
    sun_data.angle = math.radians(3)
    sun_data.color = (1.0, 0.96, 0.90)
    sun = bpy.data.objects.new("Light_Sun_Key", sun_data)
    link(sun)
    sun.rotation_euler = Euler((math.radians(50), 0, math.radians(-50)), 'XYZ')
    sun.location = (-1.5, -1.5, 2.2)

    # 正面柔光补光
    fill_data = bpy.data.lights.new("Light_Fill_AreaData", 'AREA')
    fill_data.energy = 150.0
    fill_data.size = 1.8
    fill_data.color = (0.85, 0.90, 1.0)
    fill = bpy.data.objects.new("Light_Fill_Area", fill_data)
    link(fill)
    fill.location = (0.2, -1.9, 1.9)
    fill.rotation_euler = Euler((math.radians(-55), 0, 0), 'XYZ')

    # 天花板柔和环境光，避免死角死黑
    ceil_data = bpy.data.lights.new("Light_Ceiling_SoftData", 'AREA')
    ceil_data.energy = 90.0
    ceil_data.size = 1.4
    ceil_data.color = (0.95, 0.96, 1.0)
    ceil = bpy.data.objects.new("Light_Ceiling_Soft", ceil_data)
    link(ceil)
    ceil.location = (0.0, 0.0, 2.42)
    ceil.rotation_euler = Euler((0, 0, 0), 'XYZ')

    # 右墙洗墙灯：消除死黑墙面（迭代2/4发现右侧墙全黑）
    wash_data = bpy.data.lights.new("Light_WallWash_AreaData", 'AREA')
    wash_data.energy = 80.0
    wash_data.size = 1.6
    wash_data.color = (0.88, 0.92, 1.0)
    wash = bpy.data.objects.new("Light_WallWash_Area", wash_data)
    link(wash)
    wash.location = (1.5, 0.4, 2.05)
    aim_dir = Vector((2.05, 0.9, 1.1)) - Vector((1.5, 0.4, 2.05))
    wash.rotation_euler = aim_dir.to_track_quat('-Z', 'Y').to_euler()

    world = bpy.data.worlds.new("World_LabAmbient")
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs[0].default_value = (0.02, 0.025, 0.035, 1.0)
        bg.inputs[1].default_value = 0.35
    bpy.context.scene.world = world


# ----------------------------------------------------------------------------
# 相机
# ----------------------------------------------------------------------------
def add_camera(name, location, target, lens=42.0):
    cam_data = bpy.data.cameras.new(name)
    cam_data.lens = lens
    cam_data.clip_start = 0.02
    cam_data.clip_end = 60.0
    cam = bpy.data.objects.new(name, cam_data)
    link(cam)
    cam.location = vec(location)
    direction = vec(target) - vec(location)
    cam.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
    return cam


def build_cameras():
    # 注意：相机必须在房间内部（墙体会完全遮挡室外机位）
    overview = add_camera(
        "Cam_Overview",
        (-1.62, -1.92, 1.78),
        (0.08, 0.06, 0.82),
        lens=31)
    tabletop = add_camera(
        "Cam_Tabletop",
        (-0.55, -1.52, 1.20),
        (0.02, 0.05, 0.78),
        lens=38)
    closeup = add_camera(
        "Cam_InteractionCloseup",
        (1.02, -0.50, 1.06),
        (0.48, 0.14, 0.78),
        lens=42)
    bpy.context.scene.camera = overview


# ----------------------------------------------------------------------------
# 渲染设置与初始状态记录
# ----------------------------------------------------------------------------
def setup_render_settings(samples=CAM_SAMPLES_DEFAULT):
    sc = bpy.context.scene
    sc.render.engine = 'CYCLES'
    prefs = bpy.context.preferences.addons.get("cycles")
    if prefs:
        cprefs = prefs.preferences
        try:
            cprefs.compute_device_type = 'OPTIX'
        except Exception:
            try:
                cprefs.compute_device_type = 'CUDA'
            except Exception:
                pass
        devices_ok = False
        try:
            cprefs.refresh_devices()
            devs = cprefs.devices
            for d in devs:
                if d.type != 'CPU':
                    d.use = True
                    devices_ok = True
                else:
                    d.use = True
        except Exception:
            pass
        sc.cycles.device = 'GPU' if devices_ok else 'CPU'

    cyc = sc.cycles
    cyc.samples = samples
    try:
        cyc.use_adaptive_sampling = True
        cyc.adaptive_threshold = 0.01
    except Exception:
        pass
    try:
        cyc.use_denoising = True
        cyc.denoiser = 'OPENIMAGEDENOISE'
    except Exception:
        pass
    try:
        cyc.max_bounces = 8
        cyc.transparent_max_bounces = 12  # 保证透明盒体折射正确
    except Exception:
        pass

    sc.render.resolution_x = 1280
    sc.render.resolution_y = 720
    sc.render.resolution_percentage = 100
    sc.render.image_settings.file_format = 'PNG'
    sc.render.fps = FPS
    sc.frame_start = 1
    sc.frame_end = 120
    vt = getattr(sc.view_settings, "view_transform", "")
    for want in ('AgX', 'Filmic'):
        if want.lower() in str(vt).lower() or True:
            try:
                sc.view_settings.view_transform = want
                break
            except Exception:
                continue
    try:
        sc.view_settings.look = 'AgX - Punchy'
    except Exception:
        pass
    try:
        sc.view_settings.exposure = 0.35   # 迭代1整体欠曝的补偿
    except Exception:
        pass


def record_initial_state(cube, cyl, sph, lamp_parts):
    """把所有可交互对象的初始位姿存到自定义属性，供 Reset Scene 用。"""
    resettables = {
        PROP_CUBE_RED: (cube, tuple(RED_LOC), math.radians(12)),
        PROP_CYL_BLUE: (cyl, tuple(BLUE_LOC), 0.0),
        PROP_SPH_YELLOW: (sph, tuple(YELLOW_LOC), 0.0),
    }
    for name, (obj, loc, yaw) in resettables.items():
        obj["embodied_reset_location"] = list(loc)
        obj["embodied_reset_rotation_xyz"] = [0.0, 0.0, yaw]
        obj["embodied_kind"] = "prop"

    scene = bpy.context.scene
    scene["demo_light_on"] = True
    scene["demo_box_open"] = False
    scene["demo_red_in_box"] = False
    scene["demo_light_energy_on"] = LIGHT_ON_ENERGY
    scene["demo_bulb_emission_on"] = BULB_EMISSION_ON
    scene["demo_red_target_in_box"] = list(RED_IN_BOX_LOC)
    state = json.dumps({"light_on": True, "box_open": False, "red_in_box": False})
    scene["demo_state_json"] = state
    print("[init] initial state recorded:", state)


def dump_scene_spec(path):
    """把本脚本定义的场景布局导出为 scene_spec.json（管线中间表示 IR）。

    该 JSON 是 Blender 与 MuJoCo/Isaac 后端之间的合同：
    对象几何以 primitive 描述，位姿为世界坐标（米），z-up。
    """
    import json

    def obj_entry(name, physics, shape, dims, pos, rot=None, mass=None,
                  rgba=None, semantic=None):
        e = {
            "id": name,
            "physics": physics,
            "shape": shape,
            "dims": list(dims),          # box:(x,y,z) cyl:(r,h) sph:(r,)
            "pos": [round(v, 4) for v in pos],
        }
        if rot is not None:
            e["rot_euler_xyz"] = [round(v, 4) for v in rot]
        if mass is not None:
            e["mass_kg"] = mass
        if rgba is not None:
            e["rgba"] = list(rgba)
        if semantic:
            e["semantic"] = semantic
        return e

    t = TABLE_TOP_Z
    ox, oy, oz = BOX_OUTER
    wt = BOX_WALL
    spec = {
        "schema": "scene-spec/v0.2",
        "scene_name": "embodied_lab_tabletop",
        "units": "meters",
        "up_axis": "z",
        "gravity": [0, 0, -9.81],
        "workspace": {
            "table_top_z": t,
            "table_size_xy": [TABLE_SIZE[0], TABLE_SIZE[1]],
            "table_half_thickness": TABLE_THICK / 2,
            "table_rgba": [0.20, 0.21, 0.23, 1],
            "floor_z": 0.0,
        },
        "objects": [
            obj_entry("Sample_Plinth", "static", "box",
                      (0.09, 0.09, PLINTH_H),
                      (RED_LOC.x, RED_LOC.y, TABLE_TOP_Z + PLINTH_H / 2),
                      rgba=[0.13, 0.14, 0.16, 1], semantic=["fixture"]),
            obj_entry("Prop_Cube_Red", "dynamic", "box",
                      (CUBE_SIZE, CUBE_SIZE, CUBE_SIZE),
                      (RED_LOC.x, RED_LOC.y, RED_LOC.z),
                      rot=(0, 0, math.radians(12)), mass=0.06,
                      rgba=[0.82, 0.035, 0.03, 1],
                      semantic=["grasp_target"]),
            obj_entry("Prop_Cylinder_Blue", "dynamic", "cylinder",
                      (CYL_RADIUS, CYL_HEIGHT),
                      (BLUE_LOC.x, BLUE_LOC.y, BLUE_LOC.z),
                      mass=0.18, rgba=[0.035, 0.13, 0.80, 1],
                      semantic=["distractor"]),
            obj_entry("Prop_Sphere_Yellow", "dynamic", "sphere",
                      (SPH_RADIUS,),
                      (YELLOW_LOC.x, YELLOW_LOC.y, YELLOW_LOC.z),
                      mass=0.10, rgba=[0.92, 0.72, 0.04, 1],
                      semantic=["distractor"]),
            {
                "id": "Storage_Box",
                "physics": "static_composite",
                "semantic": ["container"],
                # 四壁 + 底的局部描述：pos 相对盒中心（世界坐标=BOX_CENTER）
                "walls": [
                    # pos 相对盒中心；墙竖立在底板（厚 wt）之上：z = wt + oz/2
                    {"size": [ox, wt, oz],           "pos": [0, -(oy - wt) / 2 - wt / 2, wt + oz / 2]},
                    {"size": [ox, wt, oz],           "pos": [0, (oy - wt) / 2 + wt / 2, wt + oz / 2]},
                    {"size": [wt, oy - 2 * wt, oz],  "pos": [-(ox - wt) / 2 - wt / 2, 0, wt + oz / 2]},
                    {"size": [wt, oy - 2 * wt, oz],  "pos": [(ox - wt) / 2 + wt / 2, 0, wt + oz / 2]},
                    {"size": [ox, oy, wt],           "pos": [0, 0, wt / 2], "role": "bottom"},
                ],
                "body_pos": [BOX_CENTER.x, BOX_CENTER.y, BOX_CENTER.z],
                "inner_zone": {
                    "pos": [BOX_CENTER.x, BOX_CENTER.y,
                            BOX_CENTER.z + 2 * wt + 0.02],
                    "size": [ox - 3 * wt, oy - 3 * wt, oz - wt],
                    "rgba": [0.15, 1.0, 0.3, 0.12],
                },
                "rgba": [0.86, 0.95, 0.97, 0.45],
            },
            obj_entry("Control_Panel_Deck", "static", "box",
                      (0.20, 0.13, 0.022),
                      (-0.38, -0.30, t + 0.075),
                      rot=(math.radians(-10), 0, math.radians(8)),
                      rgba=[0.10, 0.11, 0.125, 1], semantic=["obstacle"]),
            obj_entry("Desk_Lamp_Base", "static", "cylinder", (0.085, 0.024),
                      (-0.78, 0.26, t + 0.012),
                      rgba=[0.55, 0.56, 0.58, 1], semantic=["obstacle"]),
        ],
        "robots": [
            {
                "id": "Panda_0",
                "type": "panda",
                # 地面站立、面向 +y 任务区（出厂关节零位按地面安装设计）
                "mount": "floor",
                "base_pos": [0.22, -0.45, 0.0],
                "base_yaw_deg": 0,
            }
        ],
        "cameras": [
            {"id": "agentview", "pos": [0.26, 0.42, 1.78],
             "target_xyz": [0.17, 0.03, 0.76], "fov_deg": 42},
            {"id": "wrist", "type": "attached_to_gripper"},
        ],
        "task": {
            "name": "PutRedInBox",
            "instruction": "put the red cube into the transparent storage box",
            "type": "pick_place",
            "grasp_object": "Prop_Cube_Red",
            "goal_container": "Storage_Box",
            "success_condition": {
                "object_in_zone": "Storage_Box.inner_zone",
                "min_settle_frames": 25,
                "max_speed_mps": 0.02,
                "gripper_opened": True,
            },
            "init_randomization": {
                "grasp_object_xy_jitter_m": 0.04,
                "yaw_jitter_rad": 0.7,
            },
            "grasp_constraint_note":
                "cube edge 0.05m < PandaGripper measured max aperture 0.059m",
        },
        "provenance": {
            "source_script": os.path.basename(__file__),
            "blend_file": "embodied_lab.blend",
            "generated_by": "GLM-5.3-Flash coding agent",
        },
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(spec, fh, indent=2, ensure_ascii=False)
    print("[spec]", path)



def embed_interaction_code():
    """把 interaction.py 的内容作为文本数据块存入 .blend，实现 GUI 一键运行入口。"""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "interaction.py")
    if not os.path.isfile(path):
        print("[init] interaction.py not found yet; skip embedding")
        return
    with open(path, "r", encoding="utf-8") as fh:
        code = fh.read()
    existing = bpy.data.texts.get("interaction_embedded")
    if existing:
        bpy.data.texts.remove(existing)
    txt = bpy.data.texts.new("interaction_embedded")
    txt.write(code)
    txt.use_module = False
    print("[init] embedded interaction code into text datablock 'interaction_embedded'")


def main():
    print("=" * 60)
    print("BUILD: Embodied Lab (small futuristic lab desktop)")
    print("=" * 60)
    wipe_scene()
    build_materials()
    build_room()
    table_objs = build_table()
    cube, cyl, sph = build_props()
    hinge, lid, body = build_storage_box()
    build_desk_lamp()
    build_control_panel()
    build_lights_world()
    build_cameras()
    record_initial_state(cube, cyl, sph, None)

    scene = bpy.context.scene

    def add_passive_rb(obj_name, shape='BOX', kinematic=True):
        ob = bpy.data.objects.get(obj_name)
        if ob is None:
            return
        try:
            if scene.rigidbody_world is None:
                bpy.ops.rigidbody.world_add()
            with bpy.context.temp_override(object=ob, active_object=ob,
                                           selected_objects=[ob],
                                           selected_editable_objects=[ob]):
                bpy.ops.rigidbody.object_add()
            rb = ob.rigid_body
            rb.type = 'PASSIVE'
            rb.kinematic = kinematic     # True=跟随关键帧动画，不参与动态模拟
            rb.collision_shape = shape
            rb.friction = 0.6
            rb.restitution = 0.4
            print(f"[rb] {obj_name}: passive collision body ok ({shape})")
        except Exception as exc:
            print(f"[rb] WARNING rigidbody failed for {obj_name}: {exc}")

    add_passive_rb(PROP_CUBE_RED, 'BOX')
    add_passive_rb(PROP_CYL_BLUE, 'CYLINDER')
    add_passive_rb(PROP_SPH_YELLOW, 'SPHERE')
    add_passive_rb("Table_Top", 'BOX', kinematic=False)
    add_passive_rb("Storage_Box_Body", 'BOX', kinematic=False)

    setup_render_settings()
    scene = bpy.context.scene
    dump_scene_spec(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "spec", "scene_spec.json"))
    embed_interaction_code()

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "embodied_lab.blend")
    bpy.ops.wm.save_as_mainfile(filepath=out, compress=True)
    print("[save]", out)
    names = sorted(o.name for o in scene.objects)
    print("[objects]", len(names))
    print(json.dumps(names, indent=0))
    print("BUILD_OK")


if __name__ == "__main__":
    main()

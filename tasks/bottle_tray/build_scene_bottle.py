# -*- coding: utf-8 -*-
"""
build_scene_bottle.py — Tabletop "put the green bottle in the tray" scene builder
==================================================================================
Usage (headless):
    blender --background --factory-startup --python build_scene_bottle.py -- [--samples N] [--res WxH] [--skip-render]

Constructs the whole scene from constants AND dumps the JSON IR (scene spec)
in the same run. The IR is the single source of truth consumed by:
  * this Blender scene (visual backend)
  * mujoco_env/compile_mjcf.py  (standalone physics backend)
  * mujoco_env/task_bottle_in_tray.py (robosuite task backend)

Scene: light checkerboard tabletop, green cylinder bottle (graspable:
diameter 0.05 m < PandaGripper in-sim aperture ~0.059 m), shallow deep-blue
tray (container, opening upward, slightly in front / to the side of the
bottle), one red sphere distractor (static). Panda mounts on the FLOOR at
the table edge (NullMount — no tall pedestal), recorded in the IR.
"""

import bpy
import math
import os
import sys
import json
from mathutils import Vector, Euler

# ----------------------------------------------------------------------------
# Layout constants (meters) — mirrored into the IR below. SINGLE SOURCE.
# ----------------------------------------------------------------------------
FPS = 24
# Table height 0.62 m: a floor-mounted Panda (elbow pivot ~0.33 m, max elbow
# height ~0.65 m) can only hurdle a tabletop whose surface is BELOW its max
# elbow height; at the canonical 0.75 m the forearm inevitably tunnels
# through the table edge (found by the IK collision scan), so the scene
# uses a low work bench.
TABLE_TOP_Z = 0.62
TABLE_SIZE = (1.90, 0.95)
TABLE_THICK = 0.045

# Light checkerboard table (Blender checker node scale ~= MuJoCo texrepeat)
TABLE_CHECKER = {
    "color1": (0.58, 0.59, 0.62),
    "color2": (0.88, 0.88, 0.90),
    "cell_m": 0.095,           # ~1 checker cell edge in meters
}

# --- green bottle: cylinder, graspable -------------------------------
# diameter 0.05 m < PandaGripper measured in-sim max aperture (~0.059 m)
BOTTLE_R = 0.025
BOTTLE_H = 0.14
BOTTLE_MASS = 0.15
# placed within the Panda's comfortable reach of the floor-mounted base
# (IK collision scan: 10/10 clean solutions for all jitter corners)
BOTTLE_POS = (0.02, -0.08, TABLE_TOP_Z + BOTTLE_H / 2)

# --- shallow tray: container, opening upward -------------------------
TRAY_OUTER = (0.17, 0.13, 0.045)   # x, y, wall height above table
TRAY_WALL = 0.012                  # wall / bottom thickness
# slightly in front (-y) and to the side (+x) of the bottle
TRAY_POS = (0.20, -0.26, TABLE_TOP_Z)

# --- distractor: one red sphere (static; pitfalls #15) ---------------
SPHERE_R = 0.05
SPHERE_POS = (-0.24, 0.14, TABLE_TOP_Z + SPHERE_R)

# --- robot: Panda on the floor at the table edge (no pedestal) -------
ROBOT_BASE = (0.10, -0.50, 0.0)
ROBOT_YAW_DEG = 90.0    # face the task area (+y from the base)
# collision-free "ready" posture (FK-validated in-sim): eef ~0.39 m in
# front of the base at z~0.94, palm down — a short clean IK move to the
# grasp hover waypoint
ROBOT_HOME_JOINTS = [0.0, -0.5, 0.0, -1.4, 0.0, 2.0, 0.78]

CAM_SAMPLES_DEFAULT = 48


def vec(seq):
    return Vector((seq[0], seq[1], seq[2]))


# ----------------------------------------------------------------------------
# Low-level helpers
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


def link(obj):
    bpy.context.scene.collection.objects.link(obj)


def new_obj(name, mesh_data):
    obj = bpy.data.objects.new(name, mesh_data)
    link(obj)
    return obj


def box_mesh(name, size_x, size_y, size_z, center=(0, 0, 0)):
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


def cyl_mesh(name, radius, depth, center=(0, 0, 0), rot=(0, 0, 0), verts=64):
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


def sph_mesh(name, radius, center=(0, 0, 0), segments=48, rings=24):
    import bmesh
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=segments, v_segments=rings, radius=radius)
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    obj = new_obj(name, me)
    obj.location = vec(center)
    return obj


def apply_transforms(obj):
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)


def join_objects(objs, name):
    """Merge parts; full transforms baked first so the result origin is clean."""
    objs = [o for o in objs if o and o.name in bpy.data.objects]
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


def smooth_shade(obj):
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.shade_smooth()


# ----------------------------------------------------------------------------
# Materials
# ----------------------------------------------------------------------------
MATS = {}


def make_material(name, base_color, roughness=0.4, metallic=0.0):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(bsdf.outputs[0], out.inputs[0])
    c = base_color
    bsdf.inputs["Base Color"].default_value = (c[0], c[1], c[2], 1.0)
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Metallic"].default_value = metallic
    mat.use_backface_culling = False
    return mat


def make_checker_material(name, c1, c2, cell_m):
    """Checkerboard table top — a textured horizontal surface reads as a
    surface (planar gradient cue), not as a wall."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(bsdf.outputs[0], out.inputs[0])
    tex = nt.nodes.new("ShaderNodeTexChecker")
    tex.inputs["Color1"].default_value = (c1[0], c1[1], c1[2], 1.0)
    tex.inputs["Color2"].default_value = (c2[0], c2[1], c2[2], 1.0)
    tex.inputs["Scale"].default_value = 1.0 / (2.0 * cell_m)
    coord = nt.nodes.new("ShaderNodeTexCoord")
    mapping = nt.nodes.new("ShaderNodeMapping")
    nt.links.new(coord.outputs["Object"], mapping.inputs["Vector"])
    nt.links.new(mapping.outputs["Vector"], tex.inputs["Vector"])
    nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.35
    return mat


def build_materials():
    MATS.clear()
    MATS["Floor"] = make_material("Mat_Floor", (0.42, 0.42, 0.44), roughness=0.5)
    MATS["Wall"] = make_material("Mat_Wall", (0.66, 0.64, 0.60), roughness=0.7)
    MATS["TableMetal"] = make_material("Mat_Table_Frame", (0.14, 0.14, 0.15),
                                       roughness=0.35, metallic=0.8)
    MATS["TableTop"] = make_checker_material(
        "Mat_Table_Top_Checker",
        TABLE_CHECKER["color1"], TABLE_CHECKER["color2"],
        TABLE_CHECKER["cell_m"])
    MATS["Bottle"] = make_material("Mat_Green_Bottle", (0.02, 0.36, 0.10),
                                   roughness=0.25)
    MATS["Tray"] = make_material("Mat_Tray_Blue", (0.04, 0.09, 0.50),
                                 roughness=0.4)
    MATS["Sphere"] = make_material("Mat_Red_Sphere", (0.76, 0.04, 0.04),
                                   roughness=0.3)


def set_mat(obj, key):
    if obj and obj.data is not None:
        obj.data.materials.append(MATS[key])


# ----------------------------------------------------------------------------
# Scene elements
# ----------------------------------------------------------------------------
def build_room():
    floor = box_mesh("Room_Floor", 8.0, 8.0, 0.04, (0, 0, -0.02))
    set_mat(floor, "Floor")
    back = box_mesh("Room_Wall_Back", 8.0, 0.08, 2.6, (0, 2.2, 1.3))
    set_mat(back, "Wall")
    left = box_mesh("Room_Wall_Left", 0.08, 8.0, 2.6, (-2.2, 0, 1.3))
    set_mat(left, "Wall")
    right = box_mesh("Room_Wall_Right", 0.08, 8.0, 2.6, (2.2, 0, 1.3))
    set_mat(right, "Wall")


def build_table():
    top_z_center = TABLE_TOP_Z - TABLE_THICK / 2
    top = box_mesh("Table_Top", TABLE_SIZE[0], TABLE_SIZE[1], TABLE_THICK,
                   (0, 0, top_z_center))
    set_mat(top, "TableTop")
    lx, ly = TABLE_SIZE[0] / 2 - 0.10, TABLE_SIZE[1] / 2 - 0.10
    leg_len = TABLE_TOP_Z - TABLE_THICK
    for tag, (px, py) in {"FL": (lx, -ly), "FR": (-lx, -ly),
                          "BL": (lx, ly), "BR": (-lx, ly)}.items():
        leg = cyl_mesh(f"Table_Leg_{tag}", 0.025, leg_len,
                       (px, py, leg_len / 2), verts=24)
        set_mat(leg, "TableMetal")
    return top


def build_props():
    # green bottle (grasp target)
    bottle = cyl_mesh("GreenBottle", BOTTLE_R, BOTTLE_H, BOTTLE_POS, verts=64)
    set_mat(bottle, "Bottle")
    smooth_shade(bottle)

    # shallow tray (container): 4 walls + bottom, opening upward
    ox, oy, oz = TRAY_OUTER
    t = TRAY_WALL
    parts = []
    bottom = box_mesh("TrayPart_Bottom", ox, oy, t, (0, 0, t / 2))
    set_mat(bottom, "Tray")
    parts.append(bottom)
    walls = [
        ("Front", ox, t, oz, (0, -(oy - t) / 2 - t / 2, t + oz / 2)),
        ("Back", ox, t, oz, (0, (oy - t) / 2 + t / 2, t + oz / 2)),
        ("Left", t, oy - 2 * t, oz, (-(ox - t) / 2 - t / 2, 0, t + oz / 2)),
        ("Right", t, oy - 2 * t, oz, ((ox - t) / 2 + t / 2, 0, t + oz / 2)),
    ]
    for tag, sx, sy, sz, ctr in walls:
        w = box_mesh(f"TrayPart_Wall_{tag}", sx, sy, sz, ctr)
        set_mat(w, "Tray")
        parts.append(w)
    tray = join_objects(parts, "Tray")
    tray.location = vec(TRAY_POS)
    apply_transforms(tray)

    # red sphere distractor (static)
    sph = sph_mesh("RedSphere", SPHERE_R, SPHERE_POS)
    set_mat(sph, "Sphere")
    smooth_shade(sph)
    return bottle, tray, sph


def build_lights_world():
    sun_data = bpy.data.lights.new("Light_Sun_KeyData", 'SUN')
    sun_data.energy = 2.2
    sun_data.angle = math.radians(4)
    sun_data.color = (1.0, 0.97, 0.92)
    sun = bpy.data.objects.new("Light_Sun_Key", sun_data)
    link(sun)
    sun.rotation_euler = Euler((math.radians(46), 0, math.radians(-35)), 'XYZ')
    sun.location = (-1.2, -1.4, 2.4)

    fill_data = bpy.data.lights.new("Light_Fill_AreaData", 'AREA')
    fill_data.energy = 50.0
    fill_data.size = 2.0
    fill_data.color = (0.92, 0.94, 1.0)
    fill = bpy.data.objects.new("Light_Fill_Area", fill_data)
    link(fill)
    fill.location = (0.1, -1.8, 2.0)
    fill.rotation_euler = Euler((math.radians(-52), 0, 0), 'XYZ')

    ceil_data = bpy.data.lights.new("Light_Ceiling_SoftData", 'AREA')
    ceil_data.energy = 30.0
    ceil_data.size = 1.6
    ceil = bpy.data.objects.new("Light_Ceiling_Soft", ceil_data)
    link(ceil)
    ceil.location = (0.0, 0.0, 2.45)

    world = bpy.data.worlds.new("World_Bright")
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs[0].default_value = (0.70, 0.73, 0.78, 1.0)
        bg.inputs[1].default_value = 0.22
    bpy.context.scene.world = world


def add_camera(name, location, target, lens=40.0):
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
    overview = add_camera(
        "Cam_Overview", (-1.30, -1.75, 1.85), (0.08, 0.02, 0.74), lens=32)
    tabletop = add_camera(
        "Cam_Tabletop", (-0.72, -1.25, 1.35), (0.10, 0.03, 0.755), lens=40)
    closeup = add_camera(
        "Cam_Closeup", (0.75, -0.65, 1.15), (0.13, 0.02, 0.77), lens=42)
    bpy.context.scene.camera = overview


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
        try:
            cprefs.refresh_devices()
            for d in cprefs.devices:
                d.use = True
            sc.cycles.device = 'GPU'
        except Exception:
            sc.cycles.device = 'CPU'
    cyc = sc.cycles
    cyc.samples = samples
    try:
        cyc.use_adaptive_sampling = True
        cyc.adaptive_threshold = 0.02
    except Exception:
        pass
    try:
        cyc.use_denoising = True
    except Exception:
        pass
    sc.render.resolution_x = 1280
    sc.render.resolution_y = 720
    sc.render.resolution_percentage = 100
    sc.render.image_settings.file_format = 'PNG'
    sc.render.fps = FPS
    # Standard transform keeps task-relevant colors saturated (color
    # separation between green bottle / blue tray / red sphere matters
    # for VLA data); AgX/Filmic wash them pastel.
    for want in ('Standard',):
        try:
            sc.view_settings.view_transform = want
            break
        except Exception:
            continue
    try:
        sc.view_settings.exposure = 0.0
    except Exception:
        pass


# ----------------------------------------------------------------------------
# IR dump — the single source of truth for all backends
# ----------------------------------------------------------------------------
def dump_scene_spec(path):
    """Export the scene layout as scene_spec.json (pipeline IR).

    Contract: primitive geometry only; world-frame poses in meters, z-up.
    Consumed by compile_mjcf.py (standalone physics) and
    task_bottle_in_tray.py (robosuite task).
    """

    def obj_entry(name, physics, shape, dims, pos, rot=None, mass=None,
                  rgba=None, semantic=None, extra=None):
        e = {
            "id": name,
            "physics": physics,
            "shape": shape,
            "dims": list(dims),          # box:(x,y,z) cyl:(r,h) sph:(r,)
            "pos": [round(v, 5) for v in pos],
        }
        if rot is not None:
            e["rot_euler_xyz"] = [round(v, 5) for v in rot]
        if mass is not None:
            e["mass_kg"] = mass
        if rgba is not None:
            e["rgba"] = list(rgba)
        if semantic:
            e["semantic"] = semantic
        if extra:
            e.update(extra)
        return e

    t = TABLE_TOP_Z
    ox, oy, oz = TRAY_OUTER
    wt = TRAY_WALL
    zone_z = TRAY_POS[2] + wt + BOTTLE_H / 2   # bottle-center height standing in tray
    zone_x_half = (ox - 2 * wt) / 2 - BOTTLE_R  # whole bottle inside the opening
    zone_y_half = (oy - 2 * wt) / 2 - BOTTLE_R

    spec = {
        "schema": "scene-spec/v0.2",
        "scene_name": "bottle_tray_tabletop",
        "units": "meters",
        "up_axis": "z",
        "gravity": [0, 0, -9.81],
        "workspace": {
            "table_top_z": t,
            "table_size_xy": [TABLE_SIZE[0], TABLE_SIZE[1]],
            "table_half_thickness": TABLE_THICK / 2,
            "table_rgba": [0.77, 0.77, 0.80, 1],
            "table_texture": {
                "type": "checker",
                "rgb1": list(TABLE_CHECKER["color1"]),
                "rgb2": list(TABLE_CHECKER["color2"]),
                "texrepeat": [round(TABLE_SIZE[0] / (2 * TABLE_CHECKER["cell_m"])),
                              round(TABLE_SIZE[1] / (2 * TABLE_CHECKER["cell_m"]))],
            },
            "floor_z": 0.0,
        },
        "objects": [
            obj_entry("GreenBottle", "dynamic", "cylinder",
                      (BOTTLE_R, BOTTLE_H),
                      BOTTLE_POS, mass=BOTTLE_MASS,
                      rgba=[0.02, 0.36, 0.10, 1],
                      semantic=["grasp_target"],
                      extra={"grasp": {
                          "style": "top_down",
                          "band_z_offset_note":
                              "EE grasp target = bottle_center_z + grasp_band_z_offset; "
                              "calibrated in-sim (palm droop / fingertip offset)",
                          "grasp_band_z_offset": 0.04,
                      }}),
            obj_entry("RedSphere", "static", "sphere",
                      (SPHERE_R,), SPHERE_POS,
                      rgba=[0.76, 0.04, 0.04, 1],
                      semantic=["distractor"],
                      extra={"note": "static arena geom: dynamic distractors were "
                                     "observed to tunnel during settle (pitfall #15); "
                                     "a static distractor never enters the dynamics"}),
            {
                "id": "Tray",
                "physics": "static_composite",
                "semantic": ["container"],
                "walls": [
                    {"size": [ox, wt, oz], "pos": [0, -(oy - wt) / 2 - wt / 2, wt + oz / 2]},
                    {"size": [ox, wt, oz], "pos": [0, (oy - wt) / 2 + wt / 2, wt + oz / 2]},
                    {"size": [wt, oy - 2 * wt, oz], "pos": [-(ox - wt) / 2 - wt / 2, 0, wt + oz / 2]},
                    {"size": [wt, oy - 2 * wt, oz], "pos": [(ox - wt) / 2 + wt / 2, 0, wt + oz / 2]},
                    {"size": [ox, oy, wt], "pos": [0, 0, wt / 2], "role": "bottom"},
                ],
                "body_pos": [TRAY_POS[0], TRAY_POS[1], TRAY_POS[2]],
                "inner_zone": {
                    "local_offset": [0, 0, round(wt + BOTTLE_H / 2, 5)],
                    "size": [round(2 * zone_x_half, 5), round(2 * zone_y_half, 5), 0.06],
                    "note": "success zone = vertical prism over the tray opening at the "
                            "bottle standing height; moves with the tray body",
                    "rgba": [0.15, 1.0, 0.3, 0.12],
                },
                "rgba": [0.04, 0.09, 0.50, 1],
            },
        ],
        "robots": [
            {
                "id": "Panda_0",
                "type": "panda",
                "mount": "floor",
                "base_pos": [ROBOT_BASE[0], ROBOT_BASE[1], ROBOT_BASE[2]],
                "base_yaw_deg": ROBOT_YAW_DEG,
                "reset_home_joints": list(ROBOT_HOME_JOINTS),
                "note": "NullMount (floor): the robot stands at the table edge, "
                        "no tall pedestal; home posture faces the task area",
            }
        ],
        "cameras": [
            {"id": "agentview", "pos": [-0.78, -0.28, 1.08],
             "target_xyz": [0.12, -0.18, 0.64], "fov_deg": 45},
            {"id": "wrist", "type": "attached_to_gripper"},
        ],
        "task": {
            "name": "PutBottleInTray",
            "instruction": "put the green bottle in the tray",
            "type": "pick_place",
            "grasp_object": "GreenBottle",
            "goal_container": "Tray",
            "success_condition": {
                "object_in_zone": "Tray.inner_zone",
                "max_speed_mps": 0.05,
                "gripper_opened": True,
                "note": "whole bottle inside the tray opening (center xy within "
                        "opening minus bottle radius), standing upright (center z "
                        "band), nearly at rest, gripper released",
            },
            "init_randomization": {
                "grasp_object_xy_jitter_m": 0.03,
                "tray_xy_jitter_m": 0.03,
                "yaw_jitter_rad": 0.7,
            },
            "grasp_constraint_note":
                "bottle diameter 0.05 m < PandaGripper in-sim max aperture 0.059 m; "
                "bottle height 0.14 m => top-down grasp of the upper section keeps "
                "the palm clear of the table, so NO plinth is required",
        },
        "provenance": {
            "source_script": os.path.basename(__file__),
            "blend_file": "bottle_tray_scene.blend",
            "generated_by": "GLM-5.3-Flash coding agent",
        },
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(spec, fh, indent=2, ensure_ascii=False)
    print("[spec]", path)


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def parse_args():
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    opts = {"samples": CAM_SAMPLES_DEFAULT, "res": None, "skip-render": False}
    i = 0
    while i < len(argv):
        if argv[i] == "--samples":
            opts["samples"] = int(argv[i + 1]); i += 2
        elif argv[i] == "--res":
            opts["res"] = argv[i + 1]; i += 2
        elif argv[i] == "--skip-render":
            opts["skip-render"] = True; i += 1
        else:
            i += 1
    return opts


def render_receipt(opts):
    out_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                           "renders", "m0"))
    os.makedirs(out_dir, exist_ok=True)
    sc = bpy.context.scene
    sc.cycles.samples = opts["samples"]
    if opts["res"]:
        w, h = opts["res"].lower().split("x")
        sc.render.resolution_x, sc.render.resolution_y = int(w), int(h)
    for tag, cam_name in (("overview", "Cam_Overview"),
                          ("tabletop", "Cam_Tabletop"),
                          ("closeup", "Cam_Closeup")):
        cam = bpy.data.objects.get(cam_name)
        if cam is None:
            continue
        sc.camera = cam
        sc.render.filepath = os.path.join(out_dir, f"{tag}.png").replace("\\", "/")
        print(f"[render] {tag} -> {sc.render.filepath}")
        bpy.ops.render.render(write_still=True)
    print("[render] M0 receipt done")


def main():
    opts = parse_args()
    print("=" * 60)
    print("BUILD: tabletop bottle-in-tray scene")
    print("=" * 60)
    wipe_scene()
    build_materials()
    build_room()
    build_table()
    bottle, tray, sph = build_props()
    build_lights_world()
    build_cameras()
    setup_render_settings(opts["samples"])

    here = os.path.dirname(os.path.abspath(__file__))
    dump_scene_spec(os.path.join(here, "spec", "scene_spec.json"))

    out = os.path.join(here, "bottle_tray_scene.blend")
    bpy.ops.wm.save_as_mainfile(filepath=out, compress=True)
    print("[save]", out)

    if not opts["skip-render"]:
        render_receipt(opts)
    print("BUILD_OK")


if __name__ == "__main__":
    main()

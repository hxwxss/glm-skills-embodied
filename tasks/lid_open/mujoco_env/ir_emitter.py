# -*- coding: utf-8 -*-
"""
ir_emitter.py -- shared IR -> MJCF emitters (single geometry source)
====================================================================

Both MuJoCo backends compile the IR through this module:

  * compile_mjcf.py  : standalone scene (M1 physics self-check, no robot)
  * task_open_lid.py : robosuite arena (M2+, robot injected by robosuite)

Everything is ElementTree based; only primitive geoms (box / cylinder /
sphere) are ever emitted, so contacts stay tractable.
"""

import math
import xml.etree.ElementTree as ET


def sub(parent, tag, **attrib):
    """ElementTree.SubElement with float-friendly formatting."""
    clean = {}
    for k, v in attrib.items():
        if isinstance(v, (list, tuple)):
            clean[k] = " ".join(("%.6f" % x).rstrip("0").rstrip(".") if
                                isinstance(x, float) else str(x) for x in v)
        elif isinstance(v, float):
            clean[k] = "%.6f" % v
        else:
            clean[k] = str(v)
    return ET.SubElement(parent, tag, clean)


def euler_xyz_rad_quat(rx, ry, rz):
    cz, sz = math.cos(rz / 2), math.sin(rz / 2)
    cy, sy = math.cos(ry / 2), math.sin(ry / 2)
    cx, sx = math.cos(rx / 2), math.sin(rx / 2)
    return (cx * cy * cz + sx * sy * sz,
            sx * cy * cz - cx * sy * sz,
            cx * sy * cz + sx * cy * sz,
            cx * cy * sz - sx * sy * cz)


def add_table_material(root):
    """Dark checkered work-surface material (horizontal-plane visual cue)."""
    asset = root.find(".//asset")
    if asset is None:
        asset = sub(root, "asset")
    sub(asset, "texture", name="lab_table_tex", type="2d", builtin="checker",
        rgb1="0.17 0.18 0.21", rgb2="0.30 0.31 0.35", width="640", height="640")
    sub(asset, "material", name="lab_table_mat", texture="lab_table_tex",
        texrepeat="18 10", specular="0.5", shininess="0.4", reflectance="0.08")


def emit_lights(worldbody):
    """Key light from +x (robot shadow falls away from the task zone) +
    camera-side fill; avoid multi-light shadow mush."""
    sub(worldbody, "light", name="key_light", pos="1.7 -0.7 2.5",
        dir="-0.5 0.28 -0.85", directional="true", diffuse="1.0 1.0 1.0",
        specular="0.2 0.2 0.2", castshadow="true")
    # fill from the agentview side so the robot's camera-facing surfaces
    # (normals +y) are not pitch black in the demo video
    sub(worldbody, "light", name="front_fill", pos="0.4 1.4 2.2",
        dir="-0.02 -0.38 -0.9", directional="true", diffuse="0.38 0.38 0.42")
    sub(worldbody, "light", name="fill_light", pos="-1.5 -1.0 1.8",
        dir="0.5 0.4 -0.7", directional="true", diffuse="0.22 0.22 0.25")


def emit_static_simple(worldbody, obj, rgba_override=None):
    """Static decoration geom (box / cylinder / sphere)."""
    name = obj["id"]
    pos = obj["pos"]
    kw = {"name": name, "pos": pos}
    shape = obj["shape"]
    if shape == "box":
        kw["type"] = "box"
        kw["size"] = [d / 2 for d in obj["dims"]]
    elif shape == "cylinder":
        kw["type"] = "cylinder"
        kw["size"] = [obj["dims"][0], obj["dims"][1] / 2]
    else:
        kw["type"] = "sphere"
        kw["size"] = [obj["dims"][0]]
    if obj.get("rot_euler_xyz"):
        kw["quat"] = euler_xyz_rad_quat(*obj["rot_euler_xyz"])
    kw["rgba"] = rgba_override or obj.get("rgba", [0.7, 0.7, 0.72, 1.0])
    kw["group"] = 1      # robosuite convention: 0=collision-only, 1=visual
    sub(worldbody, "geom", **kw)


def BOTTOM_TO_RIM_TOP_Z(box_obj):
    """Height of the wall rim top above the box body origin (from IR)."""
    return box_obj["walls"][0]["pos"][2] + box_obj["walls"][0]["size"][2]


def emit_lid_box(worldbody, box_obj, offset_xy=(0.0, 0.0)):
    """Compile the hinged-lid storage box from the IR.

    Emits:
      * LidBox  : static composite body (bottom + 4 walls, rim highlight
                  strips as visual-only geoms)
      * Lid     : dynamic body anchored at the hinge, with one hinge joint
                  (range/damping from IR) -- a real articulation, no welding.
    `offset_xy` shifts the whole assembly (episode jitter); both bodies get
    the same shift so box and lid never separate.
    """
    bx, by, bz = box_obj["body_pos"]
    bx += offset_xy[0]
    by += offset_xy[1]
    rgba = box_obj.get("rgba")
    bottom_rgba = box_obj.get("bottom_rgba", rgba)
    rim_rgba = box_obj.get("rim_rgba")

    box_body = sub(worldbody, "body", name=box_obj["id"], pos=(bx, by, bz))
    for wall in box_obj["walls"]:
        is_bottom = wall.get("role") == "bottom"
        gkw = {"name": "box_%s" % wall["name"], "type": "box",
               "pos": wall["pos"], "size": wall["size"],
               "rgba": bottom_rgba if is_bottom else rgba, "group": 1}
        if is_bottom:
            gkw["friction"] = "0.7 0.01 0.0005"
        sub(box_body, "geom", **gkw)

    # visual-only orange rim strips on the top edges (readability)
    if rim_rgba is not None:
        bottom = next(w for w in box_obj["walls"] if w.get("role") == "bottom")
        wx_full = bottom["size"][0] * 2
        wy_full = bottom["size"][1] * 2
        top_z = BOTTOM_TO_RIM_TOP_Z(box_obj)
        strip = 0.018
        rims = [("Front", 0.0, -(wy_full / 2 - strip / 2 - 0.001)),
                ("Back", 0.0, wy_full / 2 - strip / 2 - 0.001),
                ("Left", -(wx_full / 2 - strip / 2 - 0.001), 0.0),
                ("Right", wx_full / 2 - strip / 2 - 0.001, 0.0)]
        for rname, rx, ry in rims:
            along_x = ry != 0.0
            sx = wx_full if along_x else strip
            sy = strip if along_x else wy_full
            sub(worldbody, "geom", name="box_rim_%s" % rname, type="box",
                pos=(bx + rx, by + ry, bz + top_z + 0.002),
                size=(sx / 2, sy / 2, 0.006), rgba=rim_rgba,
                contype="0", conaffinity="0", group="2")

    # ---- hinged lid ----------------------------------------------------
    lid = box_obj["hinged_lid"]
    hx, hy, hz = lid["hinge_body_pos"]
    hx += offset_xy[0]
    hy += offset_xy[1]
    lid_body = sub(worldbody, "body", name=lid["id"], pos=(hx, hy, hz))
    sub(lid_body, "joint", name=lid["joint"], type="hinge", axis=lid["axis"],
        range=lid["range_rad"], damping=lid["damping"])
    for g in lid["geoms"]:
        # robosuite compiles with inertiagrouprange="0 0": only group-0 geoms
        # contribute mass/inertia.  Dynamic bodies therefore get a collision
        # geom (group 0, carries mass) + a visual twin (group 1, no contact).
        ckw = {"name": g["name"], "type": g["shape"], "pos": g["pos"],
               "size": g["size"], "mass": g["mass"]}
        if g.get("quat_euler_xyz_deg"):
            ckw["quat"] = euler_xyz_rad_quat(
                *[math.radians(a) for a in g["quat_euler_xyz_deg"]])
        if g.get("friction"):
            ckw["friction"] = "%.3f %.4f %.6f" % tuple(g["friction"])
        if g["shape"] == "cylinder":
            ckw["size"] = [g["size"][0], g["size"][1]]
        sub(lid_body, "geom", **ckw)
        vkw = {"name": g["name"] + "_vis", "type": g["shape"],
               "pos": g["pos"], "size": g["size"], "rgba": g["rgba"],
               "group": 1, "contype": 0, "conaffinity": 0}
        if g.get("quat_euler_xyz_deg"):
            vkw["quat"] = euler_xyz_rad_quat(
                *[math.radians(a) for a in g["quat_euler_xyz_deg"]])
        if g["shape"] == "cylinder":
            vkw["size"] = [g["size"][0], g["size"][1]]
        sub(lid_body, "geom", **vkw)


def emit_camera(worldbody, cam):
    """pos/target camera spec -> MuJoCo xyaxes camera (no quat ambiguity)."""
    pos = [float(v) for v in cam["pos"]]
    tgt = [float(v) for v in cam["target_xyz"]]
    fwd = [tgt[i] - pos[i] for i in range(3)]
    n = math.sqrt(sum(v * v for v in fwd))
    fwd = [v / n for v in fwd]
    # right = fwd x up_world (up_world = z); up = right x fwd
    right = [fwd[1], -fwd[0], 0.0]
    nr = math.sqrt(sum(v * v for v in right))
    right = [v / nr for v in right]
    up = [right[1] * fwd[2] - right[2] * fwd[1],
          right[2] * fwd[0] - right[0] * fwd[2],
          right[0] * fwd[1] - right[1] * fwd[0]]
    sub(worldbody, "camera", name=cam["id"], pos=pos,
        xyaxes=right + up, fovy=cam.get("fov_deg", 45))

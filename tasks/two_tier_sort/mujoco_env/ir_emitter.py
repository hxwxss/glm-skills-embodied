# -*- coding: utf-8 -*-
"""
ir_emitter.py -- shared IR -> MJCF emitters (single geometry source)
====================================================================

Both MuJoCo backends compile the IR through this module:

  * compile_mjcf.py    : standalone scene (M1 physics self-check, no robot)
  * task_two_tier.py   : robosuite arena (M2+, robot injected by robosuite)

Everything is ElementTree based; only primitive geoms (box / cylinder /
sphere) are ever emitted, so contacts stay tractable.

Dynamic sub-bodies (Lid, Drawer) get the dual-geom pattern: a group-0
collision geom carrying the mass + a group-1 visual twin.  robosuite
compiles with inertiagrouprange="0 0" (only group-0 geoms contribute mass),
while group-0 hulls are hidden from renders.
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


def add_render_quality(root):
    """Offscreen render quality: MSAA anti-aliasing, high-resolution
    shadows, gradient skybox background.  Safe to call on either a
    standalone <mujoco> root or the robosuite arena root."""
    visual = root.find("visual")
    if visual is None:
        visual = sub(root, "visual")
    if visual.find("quality") is None:
        sub(visual, "quality", shadowsize="4096", offsamples="8")
    sub(visual, "global", offwidth="1000", offheight="760")
    asset = root.find(".//asset")
    if asset is None:
        asset = sub(root, "asset")
    if root.find(".//texture[@name='skybox_tex']") is None:
        sub(asset, "texture", name="skybox_tex", type="skybox",
            builtin="gradient", rgb1="0.30 0.38 0.52",
            rgb2="0.88 0.90 0.93", width="512", height="512")
        sub(asset, "material", name="skybox_mat", texture="skybox_tex")


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
        specular="0.25 0.25 0.25", castshadow="true")
    sub(worldbody, "light", name="front_fill", pos="0.4 1.4 2.2",
        dir="-0.02 -0.38 -0.9", directional="true", diffuse="0.42 0.42 0.46")
    sub(worldbody, "light", name="fill_light", pos="-1.5 -1.0 1.8",
        dir="0.5 0.4 -0.7", directional="true", diffuse="0.24 0.24 0.27")
    sub(worldbody, "light", name="warm_rim", pos="0.3 -1.2 1.9",
        dir="0.0 0.35 -0.6", directional="true", diffuse="0.20 0.17 0.13")


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


def emit_free_cube(worldbody, obj, settle_gap=0.002):
    """Standalone-scene graspable cube: free joint + single box geom.

    Placed with a small gap above the support so the first integration
    steps cannot tunnel (settle-phase penetration pitfall); it settles
    during the physics self-check."""
    px, py, pz = obj["pos"]
    body = sub(worldbody, "body", name=obj["id"], pos=(px, py, pz + settle_gap))
    sub(body, "freejoint", name=obj["id"] + "_free")
    kw = {"name": obj["id"] + "_geom", "type": "box",
          "size": [d / 2 for d in obj["dims"]], "mass": obj["mass"],
          "rgba": obj.get("rgba", [0.7, 0.7, 0.72, 1.0]), "group": 1}
    if obj.get("friction"):
        kw["friction"] = "%.3f %.4f %.6f" % tuple(obj["friction"])
    sub(body, "geom", **kw)


def _dynamic_geom(body, g, suffix=""):
    """Emit one dynamic-body geom pair: group-0 collision (carries mass,
    no color) + group-1 visual twin (no contact)."""
    ckw = {"name": g["name"] + suffix, "type": g["shape"], "pos": g["pos"],
           "size": g["size"], "mass": g["mass"]}
    if g.get("quat_euler_xyz_deg"):
        ckw["quat"] = euler_xyz_rad_quat(
            *[math.radians(a) for a in g["quat_euler_xyz_deg"]])
    if g.get("friction"):
        ckw["friction"] = "%.3f %.4f %.6f" % tuple(g["friction"])
    if g["shape"] == "cylinder":
        ckw["size"] = [g["size"][0], g["size"][1]]
    sub(body, "geom", **ckw)
    vkw = {"name": g["name"] + "_vis" + suffix, "type": g["shape"],
           "pos": g["pos"], "size": g["size"], "rgba": g["rgba"],
           "group": 1, "contype": 0, "conaffinity": 0}
    if g.get("quat_euler_xyz_deg"):
        vkw["quat"] = euler_xyz_rad_quat(
            *[math.radians(a) for a in g["quat_euler_xyz_deg"]])
    if g["shape"] == "cylinder":
        vkw["size"] = [g["size"][0], g["size"][1]]
    sub(body, "geom", **vkw)


def emit_tier_box(worldbody, box_obj, offset_xy=(0.0, 0.0)):
    """Compile the two-tier storage box from the IR.

    Emits (all world-anchored so joints stay parents=world):
      * TierBox : static composite body (housing + upper compartment walls;
                  orange rim strips as visual-only geoms)
      * Lid     : dynamic body on a hinge joint at the box back-top edge
      * Drawer  : dynamic body on a slide joint under the housing top plate
    `offset_xy` shifts the whole assembly (episode jitter); all three bodies
    get the same shift so nothing separates.
    """
    bx, by, bz = box_obj["body_pos"]
    bx += offset_xy[0]
    by += offset_xy[1]
    rgba = box_obj.get("rgba")
    floor_rgba = box_obj.get("floor_rgba", rgba)
    rim_rgba = box_obj.get("rim_rgba")

    box_body = sub(worldbody, "body", name=box_obj["id"], pos=(bx, by, bz))
    for wall in box_obj["walls"]:
        role = wall.get("role")
        gkw = {"name": wall["name"], "type": "box",
               "pos": wall["pos"], "size": wall["size"],
               "rgba": floor_rgba if role else rgba, "group": 1}
        if role == "floor":
            gkw["friction"] = "0.5 0.008 0.0002"   # drawer slides on it
        sub(box_body, "geom", **gkw)

    # visual-only orange rim strips on the upper wall top edges (box-local,
    # so they follow the assembly jitter)
    if rim_rgba is not None:
        top_z = max(w["pos"][2] + w["size"][2] for w in box_obj["walls"])
        hx = box_obj["walls"][0]["size"][0]        # housing x half
        hy = box_obj["walls"][0]["size"][1]        # housing y half
        strip = 0.014
        rims = [("Front", 0.0, -(hy - strip / 2 - 0.001), hx, strip),
                ("Back", 0.0, hy - strip / 2 - 0.001, hx, strip),
                ("Left", -(hx - strip / 2 - 0.001), 0.0, strip, hy),
                ("Right", hx - strip / 2 - 0.001, 0.0, strip, hy)]
        for rname, rx, ry, sx, sy in rims:
            sub(box_body, "geom", name="tier_rim_%s" % rname, type="box",
                pos=(rx, ry, top_z + 0.002),
                size=(sx / 2, sy / 2, 0.006), rgba=rim_rgba,
                contype="0", conaffinity="0", group="2")

    # ---- hinged lid (upper tier) ----------------------------------------
    lid = box_obj["hinged_lid"]
    lx, ly, lz = lid["hinge_body_pos"]
    lx += offset_xy[0]
    ly += offset_xy[1]
    lid_body = sub(worldbody, "body", name=lid["id"], pos=(lx, ly, lz))
    sub(lid_body, "joint", name=lid["joint"], type="hinge", axis=lid["axis"],
        range=lid["range_rad"], damping=lid["damping"])
    for g in lid["geoms"]:
        _dynamic_geom(lid_body, g)

    # ---- drawer (lower tier) ---------------------------------------------
    dr = box_obj["drawer"]
    dx, dy, dz = dr["body_pos"]
    dx += offset_xy[0]
    dy += offset_xy[1]
    dr_body = sub(worldbody, "body", name=dr["id"], pos=(dx, dy, dz))
    sub(dr_body, "joint", name=dr["joint"], type="slide", axis=dr["axis"],
        range=dr["range_m"], damping=dr["damping"],
        frictionloss=dr.get("frictionloss", 0.0))
    for g in dr["geoms"]:
        _dynamic_geom(dr_body, g)


def emit_camera(worldbody, cam):
    """pos/target camera spec -> MuJoCo xyaxes camera (no quat ambiguity)."""
    pos = [float(v) for v in cam["pos"]]
    tgt = [float(v) for v in cam["target_xyz"]]
    fwd = [tgt[i] - pos[i] for i in range(3)]
    n = math.sqrt(sum(v * v for v in fwd))
    fwd = [v / n for v in fwd]
    right = [fwd[1], -fwd[0], 0.0]
    nr = math.sqrt(sum(v * v for v in right))
    right = [v / nr for v in right]
    up = [right[1] * fwd[2] - right[2] * fwd[1],
          right[2] * fwd[0] - right[0] * fwd[2],
          right[0] * fwd[1] - right[1] * fwd[0]]
    sub(worldbody, "camera", name=cam["id"], pos=pos,
        xyaxes=right + up, fovy=cam.get("fov_deg", 45))

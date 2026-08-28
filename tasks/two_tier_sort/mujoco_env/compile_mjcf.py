# -*- coding: utf-8 -*-
"""
compile_mjcf.py -- M1: IR -> standalone MuJoCo MJCF (robot-free)
================================================================

Compiles spec/scene_spec.json into generated/tier_scene.xml: floor, lab
table, static decorations, the two-tier box (lid + drawer articulations)
and the two free cubes.  The standalone scene is the artifact for the M1
physics self-check (settle / no-penetration / flip-and-hold / slide-and-hold).

Usage:
    python compile_mjcf.py [--spec ../spec/scene_spec.json] [--out generated]
"""

import argparse
import json
import os
import xml.etree.ElementTree as ET

import ir_emitter

HERE = os.path.dirname(os.path.abspath(__file__))


def compile_xml(spec):
    root = ET.Element("mujoco", model=spec["scene_name"])
    ir_emitter.sub(root, "compiler", angle="radian", autolimits="true")
    ir_emitter.sub(root, "option", timestep="0.002", gravity="0 0 -9.81")
    extent = max(spec["workspace"]["table_size_xy"]) * 1.4
    ir_emitter.sub(root, "statistic", center="0.1 0 0.55", meansize="0.04",
                   extent="%.2f" % extent)
    visual = ir_emitter.sub(root, "visual")
    ir_emitter.sub(visual, "headlight", ambient="0.45 0.45 0.5",
                   diffuse="0.65 0.65 0.7", specular="0.15 0.15 0.15")
    ir_emitter.add_render_quality(root)

    ir_emitter.add_table_material(root)
    asset = root.find(".//asset")
    ir_emitter.sub(asset, "texture", name="floor_tex", type="2d",
                   builtin="checker", rgb1="0.26 0.27 0.29",
                   rgb2="0.33 0.34 0.36", width="512", height="512")
    ir_emitter.sub(asset, "material", name="floor_mat", texture="floor_tex",
                   texrepeat="6 6", reflectance="0.12", shininess="0.4")

    world = ir_emitter.sub(root, "worldbody")
    ws = spec["workspace"]
    ir_emitter.emit_lights(world)
    ir_emitter.sub(world, "geom", name="room_floor", type="plane",
                   size="5 5 0.05", pos="0 0 %s" % ws.get("floor_z", 0.0),
                   material="floor_mat", group=1)

    hx, hy = ws["table_size_xy"][0] / 2, ws["table_size_xy"][1] / 2
    ht = ws["table_half_thickness"]
    tz = ws["table_top_z"]
    ir_emitter.sub(world, "geom", name="table_top", type="box",
                   pos=(0.0, 0.0, tz - ht), size=(hx, hy, ht),
                   material="lab_table_mat", friction="0.6 0.008 0.0002",
                   group=1)
    leg_half = (tz - 2 * ht - ws.get("floor_z", 0.0)) / 2
    for i, (lx, ly) in enumerate([(-hx + 0.09, -hy + 0.09), (hx - 0.09, -hy + 0.09),
                                  (-hx + 0.09, hy - 0.09), (hx - 0.09, hy - 0.09)]):
        ir_emitter.sub(world, "geom", name="table_leg_%d" % i, type="cylinder",
                       pos=(lx, ly, leg_half), size=(0.021, leg_half),
                       rgba="0.09 0.095 0.105 1")

    for obj in spec["objects"]:
        if obj["physics"] == "static":
            ir_emitter.emit_static_simple(world, obj)
        elif obj["physics"] == "static_composite":
            ir_emitter.emit_tier_box(world, obj)
        elif obj["physics"] == "free":
            ir_emitter.emit_free_cube(world, obj)

    for cam in spec["cameras"]:
        ir_emitter.emit_camera(world, cam)

    ir_emitter.sub(root, "actuator")
    return ET.tostring(root, encoding="unicode")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default=os.path.join(here, "..", "spec",
                                                   "scene_spec.json"))
    ap.add_argument("--out", default=os.path.join(here, "generated"))
    args = ap.parse_args()

    with open(args.spec, encoding="utf-8") as fh:
        spec = json.load(fh)

    xml = compile_xml(spec)
    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, "tier_scene.xml")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(xml)
    print("[m1] MJCF written ->", out_path, "(%d bytes)" % len(xml))
    return out_path


if __name__ == "__main__":
    main()

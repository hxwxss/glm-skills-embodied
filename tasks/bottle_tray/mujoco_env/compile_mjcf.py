# -*- coding: utf-8 -*-
"""
compile_mjcf.py — Scene Spec IR -> MuJoCo MJCF compiler (scene-spec/v0.2)
==========================================================================

Reads spec/scene_spec.json and emits a standalone MuJoCo scene:

  * static geometry: room floor, light checkerboard table, red sphere
    distractor, shallow tray (4 walls + bottom, static body with a
    child goal-zone site)
  * dynamic bodies: green bottle (free joint)
  * lights / camera for offscreen receipts

Output:
  mujoco_env/generated/bottle_tray_scene.xml

Usage:
  python compile_mjcf.py [--spec ../spec/scene_spec.json] [--out generated]

Constraint: primitives only (box/cylinder/sphere) for physics reliability.
"""

import json
import math
import os
import argparse

SETTLE_EPS = 0.0005   # rest gap above the surface, avoids initial penetration


def euler_xyz_quat(rx, ry, rz):
    rx, ry, rz = math.radians(rx), math.radians(ry), math.radians(rz)
    cz, sz = math.cos(rz / 2), math.sin(rz / 2)
    cy, sy = math.cos(ry / 2), math.sin(ry / 2)
    cx, sx = math.cos(rx / 2), math.sin(rx / 2)
    return [
        cx * cy * cz + sx * sy * sz,
        sx * cy * cz - cx * sy * sz,
        cx * sy * cz + sx * cy * sz,
        cx * cy * sz - sx * sy * cz,
    ]


class MJCFBuilder:
    def __init__(self, spec):
        self.spec = spec
        self.lines = []

    def _w(self, indent, s):
        self.lines.append("  " * indent + s)

    def build(self):
        sp = self.spec
        self.lines = ['<mujoco model="%s">' % sp["scene_name"]]

        self._w(1, '<compiler angle="radian" autolimits="true"/>')
        self._w(1, '<option timestep="0.002" gravity="0 0 -9.81"/>')
        extent = max(sp["workspace"]["table_size_xy"]) * 1.4
        self._w(1, '<visual>')
        self._w(2, '<headlight ambient="0.45 0.45 0.5" diffuse="0.65 0.65 0.7" specular="0.15 0.15 0.15"/>')
        self._w(1, '</visual>')
        self._w(1, f'<statistic center="0.1 0 0.55" meansize="0.04" extent="{extent:.2f}"/>')

        # ---- assets: light checkerboard table from the IR -----------
        tex = sp["workspace"]["table_texture"]
        self._w(1, '<asset>')
        self._w(2, '<texture name="table_tex" type="2d" builtin="checker" '
                   'rgb1="%s" rgb2="%s" width="640" height="640"/>'
                   % (" ".join(f"{v:g}" for v in tex["rgb1"]),
                      " ".join(f"{v:g}" for v in tex["rgb2"])))
        self._w(2, '<material name="table_mat" texture="table_tex" '
                   'texrepeat="%d %d" reflectance="0.06" shininess="0.3"/>'
                   % tuple(tex["texrepeat"]))
        self._w(2, '<texture name="floor_tex" type="2d" builtin="checker" '
                   'rgb1="0.30 0.31 0.33" rgb2="0.38 0.39 0.41" width="512" height="512"/>')
        self._w(2, '<material name="floor_mat" texture="floor_tex" texrepeat="6 6" '
                   'reflectance="0.10" shininess="0.4"/>')
        self._w(1, '</asset>')

        # ---- world ---------------------------------------------------
        self._w(1, '<worldbody>')
        ws = sp["workspace"]
        self._w(2, '<light name="key_light" directional="true" '
                   'pos="0.2 -1.2 2.4" dir="0.05 0.4 -0.9" diffuse="0.72 0.72 0.72"/>')
        self._w(2, '<light name="fill_light" directional="true" '
                   'pos="-1.5 -1.5 1.8" dir="0.6 0.6 -0.5" diffuse="0.35 0.36 0.4"/>')

        tz = ws["table_top_z"]
        floor_z = ws.get("floor_z", 0.0)
        self._w(2, '<geom name="room_floor" type="plane" size="5 5 0.05" '
                   'pos="0 0 %s" material="floor_mat"/>' % floor_z)

        hx, hy = ws["table_size_xy"][0] / 2, ws["table_size_xy"][1] / 2
        hthick = ws["table_half_thickness"]
        self._w(2, '<!-- table: top surface at %.3f m, light checkerboard -->' % tz)
        self._w(2, '<geom name="table_top" type="box" '
                   'pos="0 0 %.4f" size="%.4f %.4f %.4f" material="table_mat" '
                   'friction="0.6 0.008 0.0002"/>'
                   % (tz - hthick, hx, hy, hthick))
        leg_r = 0.025
        leg_half = (tz - 2 * hthick - floor_z) / 2
        for i, (lx, ly) in enumerate([(-hx + 0.10, -hy + 0.10), (hx - 0.10, -hy + 0.10),
                                      (-hx + 0.10, hy - 0.10), (hx - 0.10, hy - 0.10)]):
            self._w(2, '<geom name="table_leg_%d" type="cylinder" pos="%.4f %.4f %.4f" '
                       'size="%.4f %.4f" rgba="0.14 0.14 0.15 1"/>'
                       % (i, lx, ly, leg_half, leg_r, leg_half))

        for obj in sp["objects"]:
            if obj["physics"] == "static_composite":
                self._emit_static_composite(obj)
            elif obj["physics"] == "static":
                self._emit_static_geom(obj)

        for obj in sp["objects"]:
            if obj["physics"] == "dynamic":
                self._emit_dynamic_body(obj)

        # agentview camera from the IR
        for cam in sp.get("cameras", []):
            if cam.get("type") == "attached_to_gripper":
                continue
            pos, tgt = cam["pos"], cam["target_xyz"]
            fwd = [tgt[i] - pos[i] for i in range(3)]
            n = math.sqrt(sum(v * v for v in fwd))
            fwd = [v / n for v in fwd]
            up_w = [0.0, 0.0, 1.0]
            right = [fwd[1] * up_w[2] - fwd[2] * up_w[1],
                     fwd[2] * up_w[0] - fwd[0] * up_w[2],
                     fwd[0] * up_w[1] - fwd[1] * up_w[0]]
            nr = math.sqrt(sum(v * v for v in right))
            right = [v / nr for v in right]
            up = [right[1] * fwd[2] - right[2] * fwd[1],
                  right[2] * fwd[0] - right[0] * fwd[2],
                  right[0] * fwd[1] - right[1] * fwd[0]]
            self._w(2, '<camera name="%s" mode="fixed" pos="%s" xyaxes="%s" fovy="%d"/>'
                       % (cam["id"], " ".join(f"{v:.4f}" for v in pos),
                          " ".join(f"{v:.6f}" for v in right + up),
                          cam.get("fov_deg", 45)))

        self._w(1, '</worldbody>')
        self._w(1, '<actuator/>')
        self.lines.append('</mujoco>')
        return "\n".join(self.lines)

    def _rgba(self, obj, alpha=1.0):
        c = obj.get("rgba") or [0.7, 0.7, 0.72]
        return "%.3f %.3f %.3f %s" % (c[0], c[1], c[2], alpha)

    def _emit_static_geom(self, obj):
        shape = obj["shape"]
        p = obj["pos"]
        if shape == "box":
            size = "%f %f %f" % tuple(d / 2 for d in obj["dims"])
            t = 'type="box"'
        elif shape == "cylinder":
            r, h = obj["dims"]
            size = "%f %f" % (r, h / 2)
            t = 'type="cylinder"'
        else:
            size = "%f" % obj["dims"][0]
            t = 'type="sphere"'
        self._w(2, '<geom name="%s" %s pos="%f %f %f" size="%s" rgba="%s"/>'
                   % (obj["id"], t, p[0], p[1], p[2], size, self._rgba(obj)))

    def _emit_static_composite(self, obj):
        bx, by, bz = obj["body_pos"]
        alpha = str(obj.get("rgba", [0, 0, 0, 1])[3]) if obj.get("rgba") else "1"
        zone = obj.get("inner_zone")
        self._w(2, '<!-- %s: shallow tray, opening upward -->' % obj["id"])
        self._w(2, '<body name="%s" pos="%f %f %f">' % (obj["id"], bx, by, bz))
        for i, wall in enumerate(obj["walls"]):
            wx, wy, wz = wall["pos"]
            role = wall.get("role", "wall%d" % i)
            self._w(3, '<geom name="%s_%s" type="box" pos="%f %f %f" '
                       'size="%f %f %f" rgba="%s"/>'
                       % (obj["id"], role, wx, wy, wz,
                          wall["size"][0] / 2, wall["size"][1] / 2, wall["size"][2] / 2,
                          self._rgba(obj, alpha)))
        if zone:
            zx, zy, zz = zone["local_offset"]
            zsx, zsy, zsz = [v / 2 for v in zone["size"]]
            zs = " ".join(f"{v:.4f}" for v in (zsx, zsy, zsz))
            self._w(3, '<site name="goal_zone" type="box" pos="%f %f %f" size="%s" '
                       'rgba="0.15 1.0 0.3 0.08" group="3"/>' % (zx, zy, zz, zs))
        self._w(2, '</body>')

    def _emit_dynamic_body(self, obj):
        p = obj["pos"]
        shape = obj["shape"]
        dim = obj["dims"]
        z_abs = p[2] + SETTLE_EPS
        if shape == "box":
            geom_t = 'type="box"'
            size = "%f %f %f" % tuple(d / 2 for d in dim)
        elif shape == "cylinder":
            r, h = dim
            geom_t = 'type="cylinder"'
            size = "%f %f" % (r, h / 2)
        else:
            geom_t = 'type="sphere"'
            size = "%f" % dim[0]
        attrs = ['name="%s"' % obj["id"], 'pos="%f %f %f"' % (p[0], p[1], z_abs)]
        if obj.get("rot_euler_xyz"):
            q = euler_xyz_quat(*[math.degrees(a) for a in obj["rot_euler_xyz"]])
            attrs.append('quat="%s"' % " ".join(f"{v:.6f}" for v in q))
        mass = obj.get("mass_kg", 0.05)
        self._w(2, '<body %s>' % " ".join(attrs))
        self._w(3, '<freejoint name="%s_free"/>' % obj["id"])
        self._w(3, '<geom name="%s" %s size="%s" mass="%.3f" rgba="%s" '
                   'friction="0.7 0.01 0.0005" condim="4"/>'
                   % (obj["id"], geom_t, size, mass, self._rgba(obj)))
        sem = ";".join(obj.get("semantic", [])) or "-"
        self._w(3, '<!-- dims: %s | semantic: %s -->'
                   % ("%s" % (size,), sem))
        self._w(2, '</body>')


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default=os.path.join(here, "..", "spec", "scene_spec.json"))
    ap.add_argument("--out", default=os.path.join(here, "generated"))
    args = ap.parse_args()

    with open(args.spec, encoding="utf-8") as fh:
        spec = json.load(fh)

    xml = MJCFBuilder(spec).build()
    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, "bottle_tray_scene.xml")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(xml)
    print("[mjcf]", out_path, "(%d bytes)" % len(xml))
    return out_path


if __name__ == "__main__":
    main()

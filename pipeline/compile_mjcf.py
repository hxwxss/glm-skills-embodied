# -*- coding: utf-8 -*-
"""
compile_mjcf.py — Scene Spec IR → MuJoCo MJCF 编译器（scene-spec/v0.2）
=======================================================================

读取 spec/scene_spec.json，生成可在 MuJoCo 中仿真的 MJCF 场景：

  * 静态几何：地板、实验桌、收纳盒四壁+底、控制面板、台灯底座
  * 动态物体：红立方体 / 蓝圆柱 / 黄球（free joint，可被机械臂推动抓取）
  * 目标区：绿框半透明盒口内区可视化 site
  * 相机 / 灯光用于离屏渲染与 viewer 查看

产物：
  mujoco_env/generated/lab_scene.xml   — standalone 场景（viewer 直接打开）

用法：
  python compile_mjcf.py [--spec spec/scene_spec.json] [--out mujoco_env/generated]

设计约束：只认 primitive 几何（box/cylinder/sphere），保证物理碰撞可靠；
mesh 资产属于后续 M2+ 的增强路径。
"""

import json
import math
import os
import argparse

CUBE_HALF_EPS = 0.0005   # 落桌间隙，避免初始穿透


def rot_z_deg(deg):
    """绕 Z 轴旋转的四元数 (w,x,y,z)。"""
    r = math.radians(deg) / 2
    return [round(math.cos(r), 6), 0, 0, round(math.sin(r), 6)]


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return [
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ]


def euler_xyz_quat(rx, ry, rz):
    rx, ry, rz = math.radians(rx), math.radians(ry), math.radians(rz)
    cz, sz = math.cos(rz/2), math.sin(rz/2)
    cy, sy = math.cos(ry/2), math.sin(ry/2)
    cx, sx = math.cos(rx/2), math.sin(rx/2)
    return [
        cx*cy*cz + sx*sy*sz,
        sx*cy*cz - cx*sy*sz,
        cx*sy*cz + sx*cy*sz,
        cx*cy*sz - sx*sy*cz,
    ]


class MJCFBuilder:
    def __init__(self, spec):
        self.spec = spec
        self.lines = []

    def _w(self, indent, s):
        self.lines.append("  " * indent + s)

    # ------------------------------------------------------------------
    def build(self):
        sp = self.spec
        self.lines = ['<mujoco model="%s">' % sp["scene_name"]]

        # ---- compiler / option -------------------------------------
        self._w(1, '<compiler angle="radian" autolimits="true"/>')
        self._w(1, '<option timestep="0.002" gravity="0 0 -9.81"/>')
        extent = max(sp["workspace"]["table_size_xy"]) * 1.4
        self._w(1, '<visual>')
        self._w(2, '<headlight ambient="0.45 0.45 0.5" diffuse="0.65 0.65 0.7" specular="0.15 0.15 0.15"/>')
        self._w(1, '</visual>')
        self._w(1, f'<statistic center="0.1 0 0.55" meansize="0.04" extent="{extent:.2f}"/>')

        # ---- asset --------------------------------------------------
        self._w(1, '<asset>')
        self._w(2, '<texture name="floor_tex" type="2d" builtin="checker" '
                   'rgb1="0.26 0.27 0.29" rgb2="0.33 0.34 0.36" width="512" height="512"/>')
        self._w(2, '<material name="floor_mat" texture="floor_tex" texrepeat="6 6" '
                   'reflectance="0.12" shininess="0.4"/>')
        self._w(1, '</asset>')

        # ---- world body ---------------------------------------------
        self._w(1, '<worldbody>')
        ws = sp["workspace"]
        self._w(2, '<light name="key_light" directional="true" '
                   'pos="0 -1.2 2.4" dir="0.25 0.35 -0.9" diffuse="0.85 0.85 0.88"/>')
        self._w(2, '<light name="fill_light" directional="true" '
                   'pos="-1.5 -1.5 1.8" dir="0.6 0.6 -0.5" diffuse="0.35 0.36 0.42"/>')

        tz = ws["table_top_z"]
        floor_under_table = ws.get("floor_z", 0.0)
        self._w(2, '<geom name="room_floor" type="plane" size="5 5 0.05" '
                   'pos="0 0 %s" material="floor_mat"/>' % floor_under_table)

        hx, hy = ws["table_size_xy"][0] / 2, ws["table_size_xy"][1] / 2
        hthick = ws["table_half_thickness"]
        self._w(2, '<!-- 实验桌：顶面高度 %.3fm -->' % tz)
        self._w(2, '<geom name="table_top" type="box" '
                   'pos="0 0 %.4f" size="%.4f %.4f %.4f" '
                   'rgba="%.3f %.3f %.3f 1" friction="0.6 0.008 0.0002"/>'
                   % (tz - hthick, hx, hy, hthick, *ws["table_rgba"][:3]))

        # 四条桌腿（金属圆柱，从地面顶到桌底）
        leg_r = 0.021
        leg_half = (tz - 2 * hthick - ws.get("floor_z", 0.0)) / 2
        for i, (lx, ly) in enumerate([(-hx+0.09, -hy+0.09), (hx-0.09, -hy+0.09),
                                      (-hx+0.09, hy-0.09), (hx-0.09, hy-0.09)]):
            self._w(2, '<geom name="table_leg_%d" type="cylinder" pos="%.4f %.4f %.4f" '
                       'size="%.4f %.4f" rgba="0.09 0.095 0.105 1"/>'
                       % (i, lx, ly, leg_half, leg_r, leg_half))

        # ---- 静态复合体 & 装饰障碍 -----------------------------------
        for obj in sp["objects"]:
            if obj["physics"] == "static_composite":
                self._emit_static_composite(obj)
            elif obj["physics"] == "static":
                self._emit_static_geom(obj)

        # 目标区可视化（半透明体 + 边框，group 2/3 无碰撞）
        task = sp["task"]
        zone = next(o for o in sp["objects"]
                    if o["id"] == task["goal_container"])["inner_zone"]
        zx, zy, zz = zone["pos"]
        zsx, zsy, zsz = [v/2 for v in zone["size"]]
        zs = " ".join(f"{v:.4f}" for v in (zsx, zsy, zsz))
        self._w(2, '<!-- 任务目标区：%s -->' % task["name"])
        self._w(2, '<site name="goal_zone" type="box" pos="%.4f %.4f %.4f" '
                   'size="%s" rgba="0.15 1.0 0.3 0.08" group="3"/>'
                   % (zx, zy, zz, zs))
        self._w(2, '<geom name="goal_zone_vis" type="box" pos="%.4f %.4f %.4f" '
                   'size="%s" rgba="0.15 0.9 0.25 0.10" contype="0" conaffinity="0" group="2"/>'
                   % (zx, zy, zz, zs))

        # ---- 动态物体 -------------------------------------------------
        for obj in sp["objects"]:
            if obj["physics"] != "dynamic":
                continue
            self._emit_dynamic_body(obj)

        self._w(1, '</worldbody>')

        # ---- actuator（standalone 版本为空；机器人由 robosuite 注入）----
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
            size = "%f %f %f" % tuple(d/2 for d in obj["dims"])
            t = 'type="box"'
        elif shape == "cylinder":
            r, h = obj["dims"]
            size = "%f %f" % (r, h/2)
            t = 'type="cylinder"'
        else:
            size = "%f" % obj["dims"][0]
            t = 'type="sphere"'
        extra = ""
        if obj.get("rot_euler_xyz"):
            q = euler_xyz_quat(*[math.degrees(a) for a in obj["rot_euler_xyz"]])
            extra = ' quat="%s"' % (" ".join(f"{v:.5f}" for v in q))
        self._w(2, '<geom name="%s" %s pos="%f %f %f"%s size="%s" rgba="%s"/>'
                   % (obj["id"], t, p[0], p[1], p[2], extra, size, self._rgba(obj)))

    def _emit_static_composite(self, obj):
        bx, by, bz = obj["body_pos"]
        alpha = str(obj.get("rgba", [0, 0, 0, 1])[3]) if obj.get("rgba") else "1"
        self._w(2, '<!-- %s -->' % obj["id"])
        self._w(2, '<body name="%s" pos="%f %f %f">' % (obj["id"], bx, by, bz))
        for i, wall in enumerate(obj["walls"]):
            wx, wy, wz = wall["pos"]
            role = wall.get("role", "wall%d" % i)
            self._w(3, '<geom name="%s_%s" type="box" pos="%f %f %f" '
                       'size="%f %f %f" rgba="%s"/>'
                       % (obj["id"], role, wx, wy, wz,
                          wall["size"][0]/2, wall["size"][1]/2, wall["size"][2]/2,
                          self._rgba(obj, alpha)))
        self._w(2, '</body>')

    def _emit_dynamic_body(self, obj):
        p = obj["pos"]
        shape = obj["shape"]
        dim = obj["dims"]
        # 初始位置取 spec 的桌面落点，只抬高 0.5mm 避免初始穿透
        z_abs = p[2] + CUBE_HALF_EPS
        if shape == "box":
            geom_t = 'type="box"'
            size = "%f %f %f" % tuple(d/2 for d in dim)
            grasp_note = 'half=%s' % [round(d/2, 4) for d in dim]
        elif shape == "cylinder":
            r, h = dim
            geom_t = 'type="cylinder"'
            size = "%f %f" % (r, h/2)
            grasp_note = 'r=%s h=%s' % (r, h)
        else:
            geom_t = 'type="sphere"'
            size = "%f" % dim[0]
            grasp_note = 'r=%s' % dim[0]

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
        self._w(3, '<!-- dims: %s | semantic: %s -->' % (grasp_note, sem))
        self._w(2, '</body>')


# ---------------------------------------------------------------------------
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
    out_path = os.path.join(args.out, "lab_scene.xml")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(xml)
    print("[mjcf]", out_path, "(%d bytes)" % len(xml))
    return out_path


if __name__ == "__main__":
    main()

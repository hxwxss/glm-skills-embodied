# -*- coding: utf-8 -*-
"""
task_put_red_in_box.py — robosuite 任务封装：PutRedInBox
=========================================================

把 scene_spec.json 定义的“未来实验室桌面”编译进 robosuite 环境：

  * TableArena：1.9m × 0.95m 实验桌，顶面 0.75m（与 Blender 场景一致）
  * Panda 立于桌面后半段 (0.22, -0.30)，面向 +y 任务区（spec 定义）
  * 动态物体：RedCube(7cm) / BlueCyl / YellowBall（free joint，可抓取/推动）
  * 静态体：收纳盒四壁+底（joints=None）、控制面板、台灯底座
  * 成功判据：红块中心进入目标区 & 低速

所有布局常量都来自 scene_spec.json（单一事实来源）。

用法：
    python task_put_red_in_box.py --reset-test 30   # M2 验收
    python task_put_red_in_box.py --snap out.png    # agentview 相机快照
"""

import os
import json
import math
import argparse

import numpy as np

from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.tasks import ManipulationTask
from robosuite.models.objects.primitive.box import BoxObject
from robosuite.models.objects.primitive.cylinder import CylinderObject
from robosuite.models.objects.primitive.ball import BallObject
from robosuite.utils.placement_samplers import (
    SequentialCompositeSampler,
    UniformRandomSampler,
)
from robosuite.utils.transform_utils import mat2quat

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SPEC = os.path.join(HERE, "..", "spec", "scene_spec.json")


def load_spec(path=DEFAULT_SPEC):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def look_at_quat(pos, target):
    """MuJoCo 相机四元数(w,x,y,z)：使相机 -Z 指向 target，Y 尽量向上。"""
    pos = np.asarray(pos, dtype=float)
    fwd = np.asarray(target, dtype=float) - pos
    fwd /= np.linalg.norm(fwd)
    right = np.cross([0.0, 0.0, 1.0], fwd)
    right /= np.linalg.norm(right)
    up = np.cross(fwd, right)
    R = np.stack([right, up, -fwd], axis=1)
    q = mat2quat(R)
    return np.array(q)


def euler_xyz_rad_quat(rx, ry, rz):
    cz, sz = math.cos(rz / 2), math.sin(rz / 2)
    cy, sy = math.cos(ry / 2), math.sin(ry / 2)
    cx, sx = math.cos(rx / 2), math.sin(rx / 2)
    return np.array([
        cx * cy * cz + sx * sy * sz,
        sx * cy * cz - cx * sy * sz,
        cx * sy * cz + sx * cy * sz,
        cx * cy * sz - sx * sy * cz,
    ])


import xml.etree.ElementTree as ET


class EmbodiedLabArena(TableArena):
    """spec 定义静态场景几何（盒壁/面板/台灯）直接挂入 worldbody."""

    def __init__(self, spec):
        self.spec = spec
        super().__init__(
            table_full_size=(spec["workspace"]["table_size_xy"][0],
                             spec["workspace"]["table_size_xy"][1], 0.04),
            table_friction=(0.6, 0.008, 0.0002),
            table_offset=(0, 0, spec["workspace"]["table_top_z"]),
            has_legs=False,
        )
        self._emit_static_geoms()

    def _geom(self, name, attrib):
        attrib = dict(attrib, name=name)
        ET.SubElement(self.worldbody, "geom", attrib)

    def _emit_static_geoms(self):
        sp = self.spec
        # 桌面/桌腿颜色统一为场景深灰,与 Blender 端一致(默认白桌无法辨认朝向)
        # 桌面棋盘纹理:有纹理的水平面可被视觉系统正确感知(否则被误读为墙)
        import xml.etree.ElementTree as _ET
        asset_el = self.root.find(".//asset")
        _ET.SubElement(asset_el, "texture", attrib={
            "name": "lab_table_tex", "type": "2d", "builtin": "checker",
            "rgb1": "0.17 0.18 0.21", "rgb2": "0.30 0.31 0.35",
            "width": "640", "height": "640"})
        _ET.SubElement(asset_el, "material", attrib={
            "name": "lab_table_mat", "texture": "lab_table_tex",
            "texrepeat": "18 10", "specular": "0.5", "shininess": "0.4",
            "reflectance": "0.08"})
        for el_name in ("table_collision", "table_visual"):
            el = self.worldbody.find(f".//geom[@name='{el_name}']")
            if el is not None:
                el.set("material", "lab_table_mat")
                el.attrib.pop("rgba", None)   # 材质优先级高于 rgba,必须移除
        # 主光源(投影产生阴影,增强空间层次);同时移除 robosuite 自带
        # 灯光,避免多灯多影把任务区盖在阴影里
        for l in self.worldbody.findall("light"):
            self.worldbody.remove(l)
        ET.SubElement(self.worldbody, "light", attrib={
            "name": "key_light", "pos": "0.3 -1.2 2.4",
            "dir": "0.02 0.42 -0.9", "directional": "true",
            "diffuse": "1.0 1.0 1.0", "specular": "0.2 0.2 0.2",
            "castshadow": "true"})
        objs = {o["id"]: o for o in sp["objects"]}
        box = next(o for o in sp["objects"] if "container" in o.get("semantic", []))
        panel, lamp = objs["Control_Panel_Deck"], objs["Desk_Lamp_Base"]
        rgba = " ".join(f"{v:g}" for v in box["rgba"])
        for o in sp["objects"]:
            if o["physics"] != "static" or o["id"] in ("Control_Panel_Deck", "Desk_Lamp_Base"):
                continue
            pos = " ".join(f"{v:.5f}" for v in o["pos"])
            col = " ".join(f"{v:g}" for v in o["rgba"])
            if o["shape"] == "sphere":
                self._geom(o["id"], {"type": "sphere", "pos": pos,
                                     "size": f"{o['dims'][0]/2:.5f}", "rgba": col})
            else:
                dims_half = " ".join(f"{v/2:.5f}" for v in o["dims"])
                self._geom(o["id"], {"type": "box", "pos": pos,
                                     "size": dims_half, "rgba": col})
        bx, by, bz = box["body_pos"]
        # 盒口四条实色描边:透明玻璃壁在俯视角下对比度过低
        _bottom = next(w for w in box["walls"] if w.get("role") == "bottom")
        ox, oy, oz = _bottom["size"]
        rim = "1.0 0.62 0.0 1"
        rims = [("Front", 0, -oy / 2), ("Back", 0, oy / 2),
                ("Left", -ox / 2, 0), ("Right", ox / 2, 0)]
        for rim_name, rx, ry in rims:
            sx = ox if rx == 0 else 0.018
            sy = oy if ry == 0 else 0.018
            self._geom(f"box_rim_{rim_name}", {
                "type": "box",
                "pos": f"{bx+rx:.5f} {by+ry:.5f} {bz+oz:.5f}",
                "size": f"{sx/2:.5f} {sy/2:.5f} 0.010",
                "rgba": rim,
            })
        for i, wall in enumerate(box["walls"]):
            wx, wy, wz = wall["pos"]
            half = [v / 2 for v in wall["size"]]
            self._geom("box_%s" % wall.get("role", f"wall{i}").capitalize(), {
                "type": "box",
                "pos": f"{bx+wx:.5f} {by+wy:.5f} {bz+wz:.5f}",
                "size": " ".join(f"{v:.5f}" for v in half),
                "rgba": rgba,
            })
        pd = [v / 2 for v in panel["dims"]]
        rot = panel.get("rot_euler_xyz", [0, 0, 0])
        q = euler_xyz_rad_quat(*[math.degrees(a) for a in rot])
        self._geom("control_panel", {
            "type": "box",
            "pos": " ".join(f"{v:.5f}" for v in panel["pos"]),
            "quat": " ".join(f"{v:.6f}" for v in q),
            "size": " ".join(f"{v:.5f}" for v in pd),
            "rgba": " ".join(f"{v:g}" for v in panel["rgba"]),
        })
        lr, lh = lamp["dims"][0] / 2, lamp["dims"][1] / 2
        self._geom("lamp_base", {
            "type": "cylinder",
            "pos": " ".join(f"{v:.5f}" for v in lamp["pos"]),
            "size": f"{lr:.5f} {lh:.5f}",
            "rgba": " ".join(f"{v:g}" for v in lamp["rgba"]),
        })



class PutRedInBox(ManipulationEnv):
    """Agent 生成的实验室桌面 + Panda + pick-place 任务."""

    def __init__(
        self,
        robots="Panda",
        spec_path=DEFAULT_SPEC,
        controller_configs=None,
        base_types=None,
        gripper_types="default",
        use_camera_obs=True,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="agentview",
        control_freq=20,
        horizon=500,
        camera_names=("agentview",),
        camera_heights=256,
        camera_widths=256,
        seed=None,
    ):
        self.spec = load_spec(spec_path)

        # ---- 从 IR 解析布局常量（单一事实来源）-------------------
        self.table_top_z = float(self.spec["workspace"]["table_top_z"])
        self.table_size_xy = tuple(self.spec["workspace"]["table_size_xy"])
        objs = {o["id"]: o for o in self.spec["objects"]}
        self.red = objs["Prop_Cube_Red"]
        self.blue = objs["Prop_Cylinder_Blue"]
        self.yellow = objs["Prop_Sphere_Yellow"]
        self.box = next(o for o in self.spec["objects"]
                        if "container" in o.get("semantic", []))
        self.panel = objs["Control_Panel_Deck"]
        self.lamp = objs["Desk_Lamp_Base"]
        zone = self.box["inner_zone"]
        self.zone_center = np.array(zone["pos"], dtype=float)
        self.zone_half = np.array(zone["size"], dtype=float) / 2
        self.red_xy_jitter = float(
            self.spec["task"]["init_randomization"]["grasp_object_xy_jitter_m"])
        self.red_yaw_jitter = float(
            self.spec["task"]["init_randomization"].get("yaw_jitter_rad", 0.7))
        self.camera_specs = {c["id"]: c for c in self.spec["cameras"]}

        # Panda 官方 default_base 是 RethinkMount(Sawyer 巨型底座),会把整机
        # 抬高约半米导致够不到桌面;floor 安装时换成 NullMount 直接落地
        mount = self.spec["robots"][0].get("mount", "floor")
        if base_types is None:
            base_types = ["NullMount" if mount == "floor" else "default"]

        super().__init__(
            robots=robots,
            controller_configs=controller_configs,
            base_types=base_types,
            gripper_types=gripper_types,
            initialization_noise=None,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=False,
            render_visual_mesh=True,
            render_gpu_device_id=-1,
            control_freq=control_freq,
            lite_physics=True,
            horizon=horizon,
            ignore_done=False,
            hard_reset=True,
            camera_names=list(camera_names),
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=False,
            camera_segmentations=None,
            renderer="mjviewer",
            renderer_config=None,
            seed=seed,
        )

    # ------------------------------------------------------------------
    def _load_model(self):
        super()._load_model()

        # ---- Arena：实验桌 + 静态场景几何 --------------------------
        self.mujoco_arena = EmbodiedLabArena(self.spec)
        self.mujoco_arena.set_origin([0, 0, 0])

        # ---- Panda：立于地面、面向 +y 任务区 ------------------------
        robot_spec = self.spec["robots"][0]
        base = np.array(robot_spec["base_pos"], dtype=float)
        self.panda_base_pos = base.copy()
        self.robots[0].robot_model.set_base_xpos(base)
        yaw = math.radians(float(robot_spec.get("base_yaw_deg", 0.0)))
        if abs(yaw) > 1e-9:
            # 让手臂初始伸向 +y（spec: base_yaw_deg）——直接写 root body 四元数
            qz = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
            root_body = self.robots[0].robot_model._elements["root_body"]
            old_q = root_body.get("quat")
            new_q = "%.8f 0 0 %.8f" % qz if False else \
                f"{qz[0]:.8f} {qz[1]:.8f} {qz[2]:.8f} {qz[3]:.8f}"
            root_body.set("quat", new_q)

        # ---- 动态物体（free joint）---------------------------------
        r = self.red
        self.red_cube = BoxObject(
            name="RedCube",
            size=(r["dims"][0] / 2,) * 3,
            rgba=tuple(r["rgba"]),
        )
        b = self.blue
        self.blue_cyl = CylinderObject(
            name="BlueCyl",
            size=(b["dims"][0], b["dims"][1] / 2),
            rgba=tuple(b["rgba"]),
        )
        # 黄球在 spec 中为 static,由 arena 静态几何呈现,不进动态对象
        self.grasp_objects = [self.red_cube, self.blue_cyl]

        # ---- 组装 task ----------------------------------------------
        self.model = ManipulationTask(
            mujoco_arena=self.mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.grasp_objects,
        )

        self._get_placement_initializer()

        # ---- agentview 相机对齐 spec（用 xyaxes 定义朝向，避免 quat 歧义）---
        cam = self.camera_specs["agentview"]
        pos = np.array(cam["pos"], dtype=float)
        target = np.array(cam["target_xyz"], dtype=float)
        fwd = target - pos
        fwd /= np.linalg.norm(fwd)
        # OpenGL/MuJoCo 相机约定：(x=right, y=up, z=backward)，需保证 right=fwd×up_world
        right = np.cross(fwd, [0.0, 0.0, 1.0])
        right /= np.linalg.norm(right)
        up = np.cross(right, fwd)

        def fmt(v):
            return " ".join("%.6f" % s for s in v)

        cam_el = self.mujoco_arena.worldbody.find("./camera[@name='agentview']")
        cam_el.set("pos", fmt(pos))
        cam_el.set("fovy", "45")
        if "quat" in cam_el.attrib:
            del cam_el.attrib["quat"]
        cam_el.set("xyaxes", fmt(np.concatenate([right, up])))

    def _get_placement_initializer(self):
        self.placement_initializer = SequentialCompositeSampler(name="TaskSampler")
        top = self.table_top_z

        # 红块：任务起点 ± 抖动（yaw 也随机化）;若下方有支撑台则落在台顶
        plinth = next((o for o in self.spec["objects"]
                       if "fixture" in o.get("semantic", [])), None)
        plinth_top = 0.0
        if plinth is not None and abs(plinth["pos"][0] - self.red["pos"][0]) < 1e-6:
            plinth_top = plinth["pos"][2] + plinth["dims"][2] / 2
        self.placement_initializer.append_sampler(
            UniformRandomSampler(
                name="RedCubeSampler",
                mujoco_objects=self.red_cube,
                x_range=[-self.red_xy_jitter, self.red_xy_jitter],
                y_range=[-self.red_xy_jitter, self.red_xy_jitter],
                rotation_axis="z",
                rotation=(-self.red_yaw_jitter, self.red_yaw_jitter),
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=(self.red["pos"][0], self.red["pos"][1],
                               plinth_top),
                z_offset=0.0,
                rng=self.rng,
            )
        )
        # 蓝柱:固定位置 distractor(黄球已静态化,由 arena 呈现)
        for obj, pos in ((self.blue_cyl, self.blue["pos"]),):
            self.placement_initializer.append_sampler(
                UniformRandomSampler(
                    name=obj.name + "Sampler",
                    mujoco_objects=obj,
                    x_range=[pos[0], pos[0]],
                    y_range=[pos[1], pos[1]],
                    rotation_axis="z", rotation=0.0,
                    ensure_object_boundary_in_range=False,
                    ensure_valid_placement=True,
                    reference_pos=(pos[0], pos[1], top),
                    z_offset=0.0,
                    rng=self.rng,
                )
            )
    # ------------------------------------------------------------------
    def _setup_references(self):
        super()._setup_references()
        self.obj_body_id = {obj.name: self.sim.model.body_name2id(obj.root_body)
                            for obj in self.grasp_objects}

    def _reset_internal(self):
        super()._reset_internal()
        if self.deterministic_reset:
            return
        placements = self.placement_initializer.sample()
        for obj_pos, obj_quat, obj in placements.values():
            # 全部为 free-joint 动态物体
            self.sim.data.set_joint_qpos(
                obj.joints[0],
                np.concatenate([np.array(obj_pos), np.array(obj_quat)]),
            )

    def reward(self, action=None):
        """稀疏奖励：任务成功 +1."""
        return float(self._check_success())

    def _red_cube_root(self):
        return next(o for o in self.grasp_objects
                    if o.name == "RedCube").root_body

    def _check_success(self, gripper_closed=False):
        """成功 = 红块中心进入目标区且速度足够低."""
        cube = np.array(self.sim.data.body_xpos[self.obj_body_id["RedCube"]])
        rel = np.abs(cube - self.zone_center)
        in_zone = bool(np.all(rel < self.zone_half * 0.95))
        try:
            vel = float(np.linalg.norm(
                self.sim.data.get_body_xvelp(self._red_cube_root())))
        except Exception:
            vel = 0.0
        return in_zone and vel < 0.05


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset-test", type=int, default=0,
                    help="run N random resets and report stats")
    ap.add_argument("--snap", default=None, help="save agentview snapshot png")
    args = ap.parse_args()

    env = PutRedInBox(has_renderer=False, use_camera_obs=bool(args.snap))
    obs = env.reset()
    print("[smoke] reset ok; action_dim=%d horizon=%d"
          % (env.action_dim, env.horizon))

    if args.snap:
        try:
            import cv2
            img = obs["agentview_image"]
            cv2.imwrite(args.snap, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            print("[snap]", args.snap, img.shape)
        except Exception as e:
            print("snapshot failed:", e)

    if args.reset_test > 0:
        cube_lo = np.full(3, np.inf)
        cube_hi = -np.full(3, np.inf)
        n_success_false_ok = True
        for i in range(args.reset_test):
            env.reset()
            p = np.array(env.sim.data.body_xpos[env.obj_body_id["RedCube"]])
            cube_lo, cube_hi = np.minimum(cube_lo, p), np.maximum(cube_hi, p)
            if env._check_success():
                print("!! reset #%d already succeeds — bad init" % i)
                n_success_false_ok = False
        print("[stats] cube pos min=%s max=%s" %
              (np.round(cube_lo, 3).tolist(), np.round(cube_hi, 3).tolist()))
        ok_bounds = all(abs(cube_lo[k] - env.red["pos"][k]) <
                        env.red_xy_jitter + 1e-6 and
                        abs(cube_hi[k] - env.red["pos"][k]) <
                        env.red_xy_jitter + 1e-6 for k in (0, 1))
        print("[%s] success stays False at init" % ("ok" if n_success_false_ok else "FAIL"))
        print("[%s] jitter bounds respected" % ("ok" if ok_bounds else "FAIL"))

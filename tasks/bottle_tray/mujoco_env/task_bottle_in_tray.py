# -*- coding: utf-8 -*-
"""
task_bottle_in_tray.py — robosuite task wrapper: PutBottleInTray
=================================================================

Compiles the IR (scene_spec.json) into a robosuite manipulation env:

  * TableArena: 1.9 m x 0.95 m light checkerboard table, top at 0.75 m
  * Panda on the FLOOR at the table edge (NullMount, spec: mount=floor)
    — no tall pedestal mount
  * Dynamic object: GreenBottle (free-joint cylinder, placement jitter)
  * Static scene geoms: shallow Tray (walls+bottom, jittered per reset via
    body_pos rewrite), red sphere distractor
  * Success: bottle center inside the tray-opening zone (moves with the
    tray), standing upright, nearly at rest, gripper released

Usage:
    python task_bottle_in_tray.py --reset-test 20   # M2 acceptance
    python task_bottle_in_tray.py --snap out.png    # agentview snapshot
"""

import os
import json
import math
import argparse

import mujoco
import sys

import numpy as np

from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.tasks import ManipulationTask
from robosuite.models.objects.primitive.cylinder import CylinderObject
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


class BottleTrayArena(TableArena):
    """Static scene geometry from the IR, emitted straight into worldbody.

    The Tray is a static body (walls + bottom + goal-zone site child) so the
    per-reset jitter only needs a body_pos rewrite.
    """

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
        # light checkerboard tabletop: textured horizontal planes are
        # perceived as surfaces (not walls) and give a perspective cue
        asset_el = self.root.find(".//asset")
        tex = sp["workspace"]["table_texture"]
        ET.SubElement(asset_el, "texture", attrib={
            "name": "lab_table_tex", "type": "2d", "builtin": "checker",
            "rgb1": " ".join(f"{v:g}" for v in tex["rgb1"]),
            "rgb2": " ".join(f"{v:g}" for v in tex["rgb2"]),
            "width": "640", "height": "640"})
        ET.SubElement(asset_el, "material", attrib={
            "name": "lab_table_mat", "texture": "lab_table_tex",
            "texrepeat": "%d %d" % tuple(tex["texrepeat"]),
            "specular": "0.4", "shininess": "0.3", "reflectance": "0.05"})
        for el_name in ("table_collision", "table_visual"):
            el = self.worldbody.find(f".//geom[@name='{el_name}']")
            if el is not None:
                el.set("material", "lab_table_mat")
                el.attrib.pop("rgba", None)   # material wins over rgba
        # single key light; drop robosuite defaults (multi-light shadows).
        # MuJoCo's DEFAULT headlight would add ~0.7 diffuse on top of the key
        # light and clip the light table albedo to white — pin it explicitly.
        vis = self.root.find("visual")
        if vis is None:
            vis = ET.SubElement(self.root, "visual")
        hl = vis.find("headlight")
        if hl is None:
            hl = ET.SubElement(vis, "headlight")
        hl.set("ambient", "0.22 0.22 0.25")
        hl.set("diffuse", "0 0 0")
        hl.set("specular", "0 0 0")
        for l in self.worldbody.findall("light"):
            self.worldbody.remove(l)
        ET.SubElement(self.worldbody, "light", attrib={
            "name": "key_light", "pos": "0.3 -1.2 2.4",
            "dir": "0.05 0.4 -0.9", "directional": "true",
            "diffuse": "0.80 0.80 0.80", "specular": "0.1 0.1 0.1",
            "castshadow": "true"})

        objs = {o["id"]: o for o in sp["objects"]}
        # static distractor: red sphere
        sph = objs["RedSphere"]
        self._geom("RedSphere", {
            "type": "sphere",
            "pos": " ".join(f"{v:.5f}" for v in sph["pos"]),
            "size": f"{sph['dims'][0]:.5f}",
            "rgba": " ".join(f"{v:g}" for v in sph["rgba"]),
        })
        # shallow tray: static body (walls + bottom + goal-zone site child)
        tray = next(o for o in sp["objects"] if "container" in o.get("semantic", []))
        bx, by, bz = tray["body_pos"]
        tray_el = ET.SubElement(self.worldbody, "body", attrib={
            "name": tray["id"], "pos": f"{bx:.5f} {by:.5f} {bz:.5f}"})
        rgba = " ".join(f"{v:g}" for v in tray["rgba"])
        for i, wall in enumerate(tray["walls"]):
            wx, wy, wz = wall["pos"]
            half = [v / 2 for v in wall["size"]]
            ET.SubElement(tray_el, "geom", attrib={
                "name": "%s_%s" % (tray["id"], wall.get("role", "wall%d" % i)),
                "type": "box",
                "pos": " ".join(f"{v:.5f}" for v in (wx, wy, wz)),
                "size": " ".join(f"{v:.5f}" for v in half),
                "rgba": rgba,
            })
        zone = tray["inner_zone"]
        zx, zy, zz = zone["local_offset"]
        zsx, zsy, zsz = [v / 2 for v in zone["size"]]
        ET.SubElement(tray_el, "site", attrib={
            "name": "goal_zone", "type": "box",
            "pos": f"{zx:.5f} {zy:.5f} {zz:.5f}",
            "size": f"{zsx:.5f} {zsy:.5f} {zsz:.5f}",
            "rgba": "0.15 1.0 0.3 0.08", "group": "3"})


class PutBottleInTray(ManipulationEnv):
    """IR-generated tabletop + floor-mounted Panda + pick-place task."""

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
        horizon=1500,
        camera_names=("agentview",),
        camera_heights=256,
        camera_widths=256,
        seed=None,
    ):
        self.spec = load_spec(spec_path)

        # ---- layout constants from the IR (single source of truth) ----
        self.table_top_z = float(self.spec["workspace"]["table_top_z"])
        self.table_size_xy = tuple(self.spec["workspace"]["table_size_xy"])
        objs = {o["id"]: o for o in self.spec["objects"]}
        self.bottle = objs["GreenBottle"]
        self.sphere = objs["RedSphere"]
        self.tray = next(o for o in self.spec["objects"]
                         if "container" in o.get("semantic", []))
        zone = self.tray["inner_zone"]
        self.zone_local_offset = np.array(zone["local_offset"], dtype=float)
        self.zone_half = np.array(zone["size"], dtype=float) / 2
        self.tray_nominal = np.array(self.tray["body_pos"], dtype=float)
        ij = self.spec["task"]["init_randomization"]
        self.bottle_xy_jitter = float(ij["grasp_object_xy_jitter_m"])
        self.tray_xy_jitter = float(ij["tray_xy_jitter_m"])
        self.yaw_jitter = float(ij.get("yaw_jitter_rad", 0.7))
        self.camera_specs = {c["id"]: c for c in self.spec["cameras"]}

        # Panda default base is the tall RethinkMount pedestal; the spec says
        # mount=floor -> NullMount so the robot stands on the floor at the
        # table edge
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

        self.mujoco_arena = BottleTrayArena(self.spec)
        self.mujoco_arena.set_origin([0, 0, 0])

        robot_spec = self.spec["robots"][0]
        base = np.array(robot_spec["base_pos"], dtype=float)
        self.panda_base_pos = base.copy()
        self.robots[0].robot_model.set_base_xpos(base)
        yaw = math.radians(float(robot_spec.get("base_yaw_deg", 0.0)))
        if abs(yaw) > 1e-9:
            qz = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
            root_body = self.robots[0].robot_model._elements["root_body"]
            root_body.set("quat",
                          f"{qz[0]:.8f} {qz[1]:.8f} {qz[2]:.8f} {qz[3]:.8f}")

        # dynamic grasp object
        b = self.bottle
        self.bottle_obj = CylinderObject(
            name="GreenBottle",
            size=(b["dims"][0], b["dims"][1] / 2),
            rgba=tuple(b["rgba"]),
        )
        self.grasp_objects = [self.bottle_obj]

        self.model = ManipulationTask(
            mujoco_arena=self.mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.grasp_objects,
        )

        self._get_placement_initializer()

        # agentview camera from the IR (xyaxes form avoids quat convention traps)
        cam = self.camera_specs["agentview"]
        pos = np.array(cam["pos"], dtype=float)
        target = np.array(cam["target_xyz"], dtype=float)
        fwd = target - pos
        fwd /= np.linalg.norm(fwd)
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
        self.placement_initializer.append_sampler(
            UniformRandomSampler(
                name="GreenBottleSampler",
                mujoco_objects=self.bottle_obj,
                x_range=[-self.bottle_xy_jitter, self.bottle_xy_jitter],
                y_range=[-self.bottle_xy_jitter, self.bottle_xy_jitter],
                rotation_axis="z",
                rotation=(-self.yaw_jitter, self.yaw_jitter),
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=(self.bottle["pos"][0], self.bottle["pos"][1], top),
                z_offset=0.0,
                rng=self.rng,
            )
        )

    # ------------------------------------------------------------------
    def _setup_references(self):
        super()._setup_references()
        self.obj_body_id = {obj.name: self.sim.model.body_name2id(obj.root_body)
                            for obj in self.grasp_objects}
        self.tray_body_id = self.sim.model.body_name2id(self.tray["id"])
        self.finger_joints = [
            j for j in range(self.sim.model.njnt)
            if "finger" in (self._joint_name(j) or "")
        ]
        self.arm_qpos_adrs, self.arm_qvel_adrs = self._arm_addresses()

    def _joint_name(self, j):
        import mujoco
        return mujoco.mj_id2name(self.sim.model._model,
                                 mujoco.mjtObj.mjOBJ_JOINT, j)

    def tray_center(self):
        """Live tray body position (includes per-reset jitter)."""
        return np.array(self.sim.data.body_xpos[self.tray_body_id])

    def bottle_pos(self):
        return np.array(self.sim.data.body_xpos[self.obj_body_id["GreenBottle"]])

    def _reset_internal(self):
        super()._reset_internal()
        if self.deterministic_reset:
            return
        # bottle via placement sampler (free joint)
        placements = self.placement_initializer.sample()
        for obj_pos, obj_quat, obj in placements.values():
            self.sim.data.set_joint_qpos(
                obj.joints[0],
                np.concatenate([np.array(obj_pos), np.array(obj_quat)]),
            )
        # tray jitter: rewrite the static body's pos, then propagate
        jx = self.rng.uniform(-self.tray_xy_jitter, self.tray_xy_jitter)
        jy = self.rng.uniform(-self.tray_xy_jitter, self.tray_xy_jitter)
        tray_bid = self.sim.model.body_name2id(self.tray["id"])
        self.sim.model.body_pos[tray_bid][:2] = \
            self.tray_nominal[:2] + np.array([jx, jy])
        # IR-defined ready home posture + fingers open: the robosuite
        # default reset crouches at floor level, a poor start for a
        # floor-mounted arm at a table edge
        home = np.array(
            self.spec["robots"][0].get("reset_home_joints"), dtype=float)
        arm_qpos, arm_qvel = self._arm_addresses()
        if home.size == len(arm_qpos):
            self.sim.data.qpos[arm_qpos] = home
            self.sim.data.qvel[arm_qvel] = 0.0
            for j in self.finger_joints:
                adr = self.sim.model.jnt_qposadr[j]
                lo, hi = self.sim.model.jnt_range[j]
                self.sim.data.qpos[adr] = hi if lo >= 0 else lo
        self.sim.forward()
        # fingers are fully open right after reset: (re)capture the open
        # reference used by the released-gripper success term
        self._grip_open_max = max(1e-6, self.gripper_opening())

    def _arm_addresses(self):
        import mujoco
        m = self.sim.model._model
        arm_qpos, arm_qvel = [], []
        for j in range(m.njnt):
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
            if name.startswith("robot0_joint"):
                arm_qpos.append(m.jnt_qposadr[j])
                arm_qvel.append(m.jnt_dofadr[j])
        return np.array(arm_qpos), np.array(arm_qvel)

    def reward(self, action=None):
        return float(self._check_success())

    def gripper_opening(self):
        """Total finger opening: finger slides are +[0,0.04] and -[0.04,0];
        their signed difference is the fingertip gap in joint units
        (sum would cancel to zero)."""
        q = 0.0
        for j in self.finger_joints:
            adr = self.sim.model.jnt_qposadr[j]
            q += abs(float(self.sim.data.qpos[adr]))
        return q

    def finger_bottle_contact(self):
        """True while any finger geom touches the bottle (being held)."""
        m = self.sim.model._model
        d = self.sim.data._data
        bottle_gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM,
                                       "GreenBottle_g0")
        for i in range(d.ncon):
            c = d.contact[i]
            if bottle_gid in (c.geom1, c.geom2):
                other = c.geom2 if c.geom1 == bottle_gid else c.geom1
                bname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY,
                                          m.geom_bodyid[other]) or ""
                if "finger" in bname:
                    return True
        return False

    def _zone_center(self):
        return self.tray_center() + self.zone_local_offset

    def _check_success(self, gripper_closed=False):
        """Bottle standing inside the tray opening, nearly at rest, released."""
        b = self.bottle_pos()
        zone = self._zone_center()
        rel = np.abs(b - zone)
        in_zone = bool(np.all(rel < self.zone_half * 0.95))
        try:
            vel = float(np.linalg.norm(
                self.sim.data.get_body_xvelp("GreenBottle")))
        except Exception:
            vel = 0.0
        # released: no finger-bottle contact (primary) and fingers near open
        released = (not self.finger_bottle_contact()) and \
            self.gripper_opening() > 0.5 * self.gripper_open_max()
        return in_zone and vel < 0.05 and released

    def gripper_open_max(self):
        """Max total finger opening measured at reset (calibrated lazily)."""
        if not hasattr(self, "_grip_open_max"):
            self._grip_open_max = max(1e-6, self.gripper_opening())
        return self._grip_open_max


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset-test", type=int, default=0)
    ap.add_argument("--snap", default=None)
    args = ap.parse_args()

    env = PutBottleInTray(has_renderer=False, use_camera_obs=bool(args.snap))
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
        bot_lo = np.full(3, np.inf)
        bot_hi = -np.full(3, np.inf)
        tray_lo = np.full(3, np.inf)
        tray_hi = -np.full(3, np.inf)
        success_false_ok = True
        on_surface_ok = True
        sep_ok = True
        for i in range(args.reset_test):
            env.reset()
            bp = env.bottle_pos()
            tp = env.tray_center()
            bot_lo, bot_hi = np.minimum(bot_lo, bp), np.maximum(bot_hi, bp)
            tray_lo, tray_hi = np.minimum(tray_lo, tp), np.maximum(tray_hi, tp)
            if env._check_success():
                print("!! reset #%d already succeeds — bad init" % i)
                success_false_ok = False
            # bottle must rest on the table right after reset
            if abs(bp[2] - (env.table_top_z + env.bottle["dims"][1] / 2)) > 0.01:
                print("!! reset #%d bottle off the table: z=%.3f" % (i, bp[2]))
                on_surface_ok = False
            # bottle must not start inside/overlapping the tray
            dx = abs(bp[0] - tp[0])
            dy = abs(bp[1] - tp[1])
            if dx < (0.17 / 2 + 0.025 + 0.005) and dy < (0.13 / 2 + 0.025 + 0.005):
                print("!! reset #%d bottle starts overlapping tray" % i)
                sep_ok = False
        print("[stats] bottle pos min=%s max=%s" %
              (np.round(bot_lo, 3).tolist(), np.round(bot_hi, 3).tolist()))
        print("[stats] tray   pos min=%s max=%s" %
              (np.round(tray_lo, 3).tolist(), np.round(tray_hi, 3).tolist()))
        j_ok = True
        for lo, hi, nom, jj in ((bot_lo, bot_hi, env.bottle["pos"], env.bottle_xy_jitter),
                                (tray_lo, tray_hi, env.tray_nominal, env.tray_xy_jitter)):
            for k in (0, 1):
                if abs(lo[k] - nom[k]) > jj + 1e-6 or abs(hi[k] - nom[k]) > jj + 1e-6:
                    j_ok = False
        print("[%s] success stays False at init" % ("ok" if success_false_ok else "FAIL"))
        print("[%s] objects rest on surfaces" % ("ok" if on_surface_ok else "FAIL"))
        print("[%s] jitter bounds respected" % ("ok" if j_ok else "FAIL"))
        print("[%s] no bottle/tray initial overlap" % ("ok" if sep_ok else "FAIL"))
        if not (success_false_ok and on_surface_ok and j_ok and sep_ok):
            sys.exit(1)
        if not (success_false_ok and on_surface_ok and j_ok and sep_ok):
            sys.exit(1)
        print("RESET_TEST_OK")

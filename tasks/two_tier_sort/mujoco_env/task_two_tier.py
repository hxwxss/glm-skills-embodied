# -*- coding: utf-8 -*-
"""
task_two_tier.py -- M2: robosuite task wrapper "TwoTierSort"
=============================================================

Compiles the scene IR into a robosuite ManipulationEnv:

  * TierArena     : spec table + static decorations + the two-tier box
                    (housing + hinged lid + drawer), all compiled through
                    ir_emitter -- same code path as the standalone M1 scene
  * Robot         : Panda on the spec mount at the table edge, facing +y
  * Articulations : the lid hinge and the drawer slide are real joints
                    living in the arena; `_reset_internal` zeroes them and
                    jitters the whole box assembly per episode (body_pos
                    writes to the freshly rebuilt MjModel)
  * Dynamic props : RedCube / BlueCube BoxObjects (free joints) placed by a
                    SequentialCompositeSampler with xy + yaw jitter
  * Success       : red cube rests in the upper-compartment zone AND blue
                    cube rests in the drawer-cavity zone (zones tracked
                    live: the tray zone moves with the drawer), both
                    near-stationary

Usage:
    python task_two_tier.py --reset-test 10     # M2 acceptance
    python task_two_tier.py --snap out.png      # agentview snapshot
"""

import argparse
import json
import math
import os
import sys

import numpy as np

from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.tasks import ManipulationTask
from robosuite.models.objects.primitive.box import BoxObject
from robosuite.utils.placement_samplers import (
    SequentialCompositeSampler,
    UniformRandomSampler,
)

import ir_emitter

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SPEC = os.path.join(HERE, "..", "spec", "scene_spec.json")


def load_spec(path=DEFAULT_SPEC):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def tier_box_obj(spec):
    return next(o for o in spec["objects"] if o["physics"] == "static_composite")


class TierArena(TableArena):
    """spec-defined static scene + the two-tier box, in the worldbody."""

    def __init__(self, spec):
        self.spec = spec
        super().__init__(
            table_full_size=(spec["workspace"]["table_size_xy"][0],
                             spec["workspace"]["table_size_xy"][1], 0.04),
            table_friction=(0.6, 0.008, 0.0002),
            table_offset=(0, 0, spec["workspace"]["table_top_z"]),
            has_legs=False,
        )
        self._customize_table()
        self._emit_scene()

    def _customize_table(self):
        # dark checkered work surface + single key light; robosuite's own
        # lights must go or shadows pile up (pitfall)
        ir_emitter.add_table_material(self.root)
        visual = self.root.find("visual")
        if visual is None:
            visual = ir_emitter.sub(self.root, "visual")
        for el in visual.findall("headlight"):
            visual.remove(el)
        ir_emitter.sub(visual, "headlight", ambient="0.35 0.35 0.4",
                       diffuse="0.45 0.45 0.5", specular="0.1 0.1 0.1")
        ir_emitter.add_render_quality(self.root)
        for el_name in ("table_collision", "table_visual"):
            el = self.worldbody.find(f".//geom[@name='{el_name}']")
            if el is not None:
                el.set("material", "lab_table_mat")
                el.attrib.pop("rgba", None)     # material wins over rgba
        for l in self.worldbody.findall("light"):
            self.worldbody.remove(l)
        ir_emitter.emit_lights(self.worldbody)

    def _emit_scene(self):
        for obj in self.spec["objects"]:
            if obj["physics"] == "static":
                ir_emitter.emit_static_simple(self.worldbody, obj)
            elif obj["physics"] == "static_composite":
                ir_emitter.emit_tier_box(self.worldbody, obj)


class TwoTierSort(ManipulationEnv):
    """Tabletop scene with a Panda and a two-tier box; task = open both
    tiers and sort the red/blue cubes into them."""

    def __init__(
        self,
        robots="Panda",
        spec_path=DEFAULT_SPEC,
        controller_configs=None,
        base_types=None,
        gripper_types="default",
        use_camera_obs=False,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="agentview",
        control_freq=20,
        horizon=4600,
        camera_names=("agentview",),
        camera_heights=256,
        camera_widths=256,
        seed=None,
    ):
        self.spec = load_spec(spec_path)

        # ---- layout constants from the IR (single source of truth) ------
        ws = self.spec["workspace"]
        self.table_top_z = float(ws["table_top_z"])
        self.tier_box = tier_box_obj(self.spec)
        self.lid_spec = self.tier_box["hinged_lid"]
        self.drawer_spec = self.tier_box["drawer"]
        self.upper_zone_c = np.array(
            self.tier_box["upper_zone_local"]["center"], dtype=float)
        self.upper_zone_h = np.array(
            self.tier_box["upper_zone_local"]["half"], dtype=float)
        self.tray_zone_c = np.array(
            self.tier_box["tray_zone_local"]["center"], dtype=float)
        self.tray_zone_h = np.array(
            self.tier_box["tray_zone_local"]["half"], dtype=float)
        self.max_speed = float(
            self.spec["task"]["success_condition"]["max_speed_m_s"])
        rnd = self.spec["task"]["init_randomization"]
        self.box_jitter = float(rnd["box_xy_jitter_m"])
        self.prop_jitter = float(rnd["prop_xy_jitter_m"])
        self.prop_yaw = float(rnd["prop_yaw_jitter_rad"])
        self.camera_specs = {c["id"]: c for c in self.spec["cameras"]}
        self.prop_specs = {o["id"]: o for o in self.spec["objects"]
                           if o.get("physics") == "free"}

        mount = self.spec["robots"][0].get("mount", "RethinkMount")
        if base_types is None:
            base_types = [mount if mount != "floor" else "NullMount"]

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

        self.mujoco_arena = TierArena(self.spec)
        self.mujoco_arena.set_origin([0, 0, 0])

        robot_spec = self.spec["robots"][0]
        base = np.array(robot_spec["base_pos"], dtype=float)
        self.panda_base_pos = base.copy()
        self.robots[0].robot_model.set_base_xpos(base)
        yaw = math.radians(float(robot_spec.get("base_yaw_deg", 0.0)))
        if abs(yaw) > 1e-9:
            qz = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
            root_body = self.robots[0].robot_model._elements["root_body"]
            root_body.set("quat", "%.8f %.8f %.8f %.8f" % qz)

        # ---- dynamic grasp props (free joints) --------------------------
        self.red_cube = BoxObject(
            name="RedCube",
            size=(self.prop_specs["RedCube"]["dims"][0] / 2,) * 3,
            rgba=tuple(self.prop_specs["RedCube"]["rgba"]),
            friction=tuple(self.prop_specs["RedCube"].get(
                "friction", [1.0, 0.005, 0.0001])),
        )
        self.blue_cube = BoxObject(
            name="BlueCube",
            size=(self.prop_specs["BlueCube"]["dims"][0] / 2,) * 3,
            rgba=tuple(self.prop_specs["BlueCube"]["rgba"]),
            friction=tuple(self.prop_specs["BlueCube"].get(
                "friction", [1.0, 0.005, 0.0001])),
        )
        self.grasp_objects = [self.red_cube, self.blue_cube]

        self.model = ManipulationTask(
            mujoco_arena=self.mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.grasp_objects,
        )
        self._get_placement_initializer()
        self._align_cameras()

    def _align_cameras(self):
        def fmt(v):
            return " ".join("%.6f" % s for s in v)

        for cam in self.spec["cameras"]:
            el = self.mujoco_arena.worldbody.find(
                "./camera[@name='%s']" % cam["id"])
            if el is None:
                continue
            pos = np.array(cam["pos"], dtype=float)
            target = np.array(cam["target_xyz"], dtype=float)
            fwd = target - pos
            fwd /= np.linalg.norm(fwd)
            right = np.cross(fwd, [0.0, 0.0, 1.0])
            right /= np.linalg.norm(right)
            up = np.cross(right, fwd)
            el.set("pos", fmt(pos))
            el.set("fovy", str(int(cam.get("fov_deg", 45))))
            if "quat" in el.attrib:
                del el.attrib["quat"]
            el.set("xyaxes", fmt(np.concatenate([right, up])))

    def _get_placement_initializer(self):
        self.placement_initializer = SequentialCompositeSampler(
            name="TaskSampler")
        for obj in self.grasp_objects:
            p = self.prop_specs[obj.name]["pos"]
            self.placement_initializer.append_sampler(
                UniformRandomSampler(
                    name=obj.name + "Sampler",
                    mujoco_objects=obj,
                    x_range=[-self.prop_jitter, self.prop_jitter],
                    y_range=[-self.prop_jitter, self.prop_jitter],
                    rotation_axis="z",
                    rotation=(-self.prop_yaw, self.prop_yaw),
                    ensure_object_boundary_in_range=False,
                    ensure_valid_placement=True,
                    reference_pos=(p[0], p[1], self.table_top_z),
                    z_offset=0.002,        # settle gap (tunneling pitfall)
                    rng=self.rng,
                )
            )

    def _setup_references(self):
        super()._setup_references()
        m = self.sim.model
        self.obj_body_id = {obj.name: m.body_name2id(obj.root_body)
                            for obj in self.grasp_objects}
        self.box_body_id = m.body_name2id(self.tier_box["id"])
        self.lid_body_id = m.body_name2id(self.lid_spec["id"])
        self.drawer_body_id = m.body_name2id(self.drawer_spec["id"])
        self.lid_bar_gid = m.geom_name2id("lid_handle_bar")
        self.drawer_bar_gid = m.geom_name2id("drawer_handle_bar")
        self.lid_qpos_adr = m.jnt_qposadr[
            m.joint_name2id(self.lid_spec["joint"])]
        self.lid_dof_adr = m.jnt_dofadr[
            m.joint_name2id(self.lid_spec["joint"])]
        self.slide_qpos_adr = m.jnt_qposadr[
            m.joint_name2id(self.drawer_spec["joint"])]
        self.slide_dof_adr = m.jnt_dofadr[
            m.joint_name2id(self.drawer_spec["joint"])]
        # pristine (unjittered) body origins, restored by hard_reset rebuild
        self.body_pos0 = {
            key: np.array(m.body_pos[bid]).copy()
            for key, bid in (("box", self.box_body_id),
                             ("lid", self.lid_body_id),
                             ("drawer", self.drawer_body_id))
        }

    def _reset_internal(self):
        super()._reset_internal()
        if not self.deterministic_reset:
            # whole box+lid+drawer assembly jitters together
            dxy = self.rng.uniform(-self.box_jitter, self.box_jitter, 2)
            m = self.sim.model
            for key in ("box", "lid", "drawer"):
                bid = {"box": self.box_body_id, "lid": self.lid_body_id,
                       "drawer": self.drawer_body_id}[key]
                m.body_pos[bid][:2] = self.body_pos0[key][:2] + dxy
            placements = self.placement_initializer.sample()
            for obj_pos, obj_quat, obj in placements.values():
                self.sim.data.set_joint_qpos(
                    obj.joints[0],
                    np.concatenate([np.array(obj_pos), np.array(obj_quat)]),
                )
                self.sim.data.set_joint_qvel(obj.joints[0], np.zeros(6))
        # both articulations start fully closed, at rest
        self.sim.data.set_joint_qpos(self.lid_spec["joint"], np.zeros(1))
        self.sim.data.set_joint_qvel(self.lid_spec["joint"], np.zeros(1))
        self.sim.data.set_joint_qpos(self.drawer_spec["joint"], np.zeros(1))
        self.sim.data.set_joint_qvel(self.drawer_spec["joint"], np.zeros(1))
        self.sim.forward()

    # ------------------------------------------------------------------
    def lid_angle(self):
        return float(self.sim.data.qpos[self.lid_qpos_adr])

    def drawer_slide(self):
        return float(self.sim.data.qpos[self.slide_qpos_adr])

    def lid_bar_pos(self):
        return np.array(self.sim.data.geom_xpos[self.lid_bar_gid])

    def drawer_bar_pos(self):
        return np.array(self.sim.data.geom_xpos[self.drawer_bar_gid])

    def hinge_world(self):
        return np.array(self.sim.data.body_xpos[self.lid_body_id])

    def prop_pos(self, name):
        bid = self.obj_body_id[name]
        return np.array(self.sim.data.body_xpos[bid])

    def prop_speed(self, name):
        obj = self.red_cube if name == "RedCube" else self.blue_cube
        try:
            return float(np.linalg.norm(
                self.sim.data.get_body_xvelp(obj.root_body)))
        except Exception:
            return 0.0

    def upper_zone_world(self):
        """(center, half) of the upper-compartment zone, live (box jitters)."""
        return (np.array(self.sim.data.body_xpos[self.box_body_id])
                + self.upper_zone_c, self.upper_zone_h)

    def tray_zone_world(self):
        """(center, half) of the drawer-cavity zone, live (drawer slides)."""
        return (np.array(self.sim.data.body_xpos[self.drawer_body_id])
                + self.tray_zone_c, self.tray_zone_h)

    def reward(self, action=None):
        return float(self._check_success())

    def _check_success(self, gripper_closed=False):
        succ = self.spec["task"]["success_condition"]
        vmax = float(succ["max_speed_m_s"])
        lid_closed = float(succ.get("lid_closed_rad", 0.06))
        drawer_closed = float(succ.get("drawer_closed_m", 0.012))
        red = self.prop_pos("RedCube")
        uc, uh = self.upper_zone_world()
        blue = self.prop_pos("BlueCube")
        tc, th = self.tray_zone_world()
        red_in = bool(np.all(np.abs(red - uc) < uh + 0.002))
        blue_in = bool(np.all(np.abs(blue - tc) < th + 0.002))
        slow = (self.prop_speed("RedCube") < vmax and
                self.prop_speed("BlueCube") < vmax)
        closed = (abs(self.lid_angle()) < lid_closed and
                  self.drawer_slide() < drawer_closed)
        return red_in and blue_in and slow and closed


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset-test", type=int, default=0)
    ap.add_argument("--snap", default=None)
    args = ap.parse_args()

    env = TwoTierSort(has_renderer=False, use_camera_obs=bool(args.snap))
    obs = env.reset()
    print("[smoke] reset ok; action_dim=%d horizon=%d joints=%s/%s"
          % (env.action_dim, env.horizon, env.lid_spec["joint"],
             env.drawer_spec["joint"]))

    if args.snap:
        try:
            import mjrender as R
            from PIL import Image
            renderer = R.make_renderer(env, height=480, width=640)
            img = R.render_snapshot(renderer, env, R.camera_fixed(env, "agentview"))
            renderer.close()
            Image.fromarray(img).save(args.snap)
            print("[snap]", args.snap)
        except Exception as e:
            print("snapshot failed:", e)

    if args.reset_test > 0:
        red_lo = np.full(3, np.inf)
        red_hi = -np.full(3, np.inf)
        blue_lo = np.full(3, np.inf)
        blue_hi = -np.full(3, np.inf)
        ok_init = True
        ok_closed = True
        for i in range(args.reset_test):
            env.reset()
            for name, lo, hi in (("RedCube", red_lo, red_hi),
                                 ("BlueCube", blue_lo, blue_hi)):
                p = env.prop_pos(name)
                lo[:] = np.minimum(lo, p)
                hi[:] = np.maximum(hi, p)
            if env._check_success():
                print("!! reset #%d already succeeds -- bad init" % i)
                ok_init = False
            if (abs(env.lid_angle()) > 1e-3 or abs(env.drawer_slide()) > 1e-3):
                print("!! reset #%d articulations not shut: lid=%.4f "
                      "drawer=%.4f" % (i, env.lid_angle(),
                                       env.drawer_slide()))
                ok_closed = False
        for name, lo, hi, spec in (("RedCube", red_lo, red_hi,
                                    env.prop_specs["RedCube"]),
                                   ("BlueCube", blue_lo, blue_hi,
                                    env.prop_specs["BlueCube"])):
            span = hi - lo
            want = 2 * env.prop_jitter + 1e-3
            ok_b = all(abs(lo[k] - spec["pos"][k]) < want and
                       abs(hi[k] - spec["pos"][k]) < want for k in (0, 1))
            print("[%s] %s jitter span=%s (want <= %s)"
                  % ("ok" if ok_b else "FAIL", name,
                     np.round(span, 4).tolist(), round(2 * env.prop_jitter, 3)))
            if not ok_b:
                ok_init = False
        print("[%s] success stays False at init"
              % ("ok" if ok_init else "FAIL"))
        print("[%s] lid and drawer shut at every reset"
              % ("ok" if ok_closed else "FAIL"))
        sys.exit(0 if ok_init and ok_closed else 1)

# -*- coding: utf-8 -*-
"""
task_open_lid.py -- M2: robosuite task wrapper "OpenBoxLid"
============================================================

Compiles the scene IR into a robosuite ManipulationEnv:

  * LidArena      : spec table + static decorations + the hinged-lid box
                    (all compiled through ir_emitter -- same code path as
                    the standalone M1 scene, zero geometry drift)
  * Robot         : Panda on the spec mount at the table edge, facing +y
  * Articulation  : the lid is a real hinge joint living in the arena;
                    `_reset_internal` zeroes it and jitters the whole
                    box+lid assembly per episode (model.body_pos writes)
  * Success       : lid angle <= lid_angle_max_rad (past vertical) and
                    near-stationary

Usage:
    python task_open_lid.py --reset-test 10     # M2 acceptance
    python task_open_lid.py --snap out.png      # agentview snapshot
"""

import argparse
import json
import math
import os

import numpy as np
import xml.etree.ElementTree as ET

from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.placement_samplers import SequentialCompositeSampler
from robosuite.utils.transform_utils import mat2quat

import ir_emitter

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SPEC = os.path.join(HERE, "..", "spec", "scene_spec.json")


def load_spec(path=DEFAULT_SPEC):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def lid_box_obj(spec):
    return next(o for o in spec["objects"] if o["physics"] == "static_composite")


class LidArena(TableArena):
    """spec-defined static scene + the hinged-lid box, in the worldbody."""

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
        # dark checkered work surface (horizontal-plane visual cue) + single
        # key light; robosuite's own lights must go or shadows pile up
        ir_emitter.add_table_material(self.root)
        # tame robosuite's strong default headlight (washes the scene out)
        visual = self.root.find("visual")
        if visual is None:
            visual = ir_emitter.sub(self.root, "visual")
        for el in visual.findall("headlight"):
            visual.remove(el)
        ir_emitter.sub(visual, "headlight", ambient="0.35 0.35 0.4",
                       diffuse="0.45 0.45 0.5", specular="0.1 0.1 0.1")
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
                ir_emitter.emit_lid_box(self.worldbody, obj)


class OpenBoxLid(ManipulationEnv):
    """Tabletop scene with a Panda and a hinged-lid box; task = flip lid open."""

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
        horizon=900,
        camera_names=("agentview",),
        camera_heights=256,
        camera_widths=256,
        seed=None,
    ):
        self.spec = load_spec(spec_path)

        # ---- layout constants from the IR (single source of truth) ------
        self.table_top_z = float(self.spec["workspace"]["table_top_z"])
        self.lid_box = lid_box_obj(self.spec)
        self.hinged = self.lid_box["hinged_lid"]
        self.joint_name = self.hinged["joint"]
        self.knob_local = np.array(self.hinged["knob_local"], dtype=float)
        succ = self.spec["task"]["success_condition"]
        self.lid_angle_max = float(succ["lid_angle_max_rad"])
        self.lid_speed_max = float(succ["lid_max_speed_rad_s"])
        self.box_jitter = float(
            self.spec["task"]["init_randomization"]["box_xy_jitter_m"])
        self.camera_specs = {c["id"]: c for c in self.spec["cameras"]}

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

        self.mujoco_arena = LidArena(self.spec)
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

        # no free-joint task objects: everything lives in the arena
        self.model = ManipulationTask(
            mujoco_arena=self.mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=[],
        )
        self._get_placement_initializer()
        self._align_cameras()

    def _align_cameras(self):
        def fmt(v):
            return " ".join("%.6f" % s for s in v)

        for cam in self.spec["cameras"]:
            if cam["id"] not in [c["id"] for c in self.spec["cameras"]]:
                continue
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
        # no dynamic objects to place; kept for parity with the pipeline
        self.placement_initializer = SequentialCompositeSampler(
            name="TaskSampler")

    def _setup_references(self):
        super()._setup_references()
        m = self.sim.model
        self.box_body_id = m.body_name2id(self.lid_box["id"])
        self.lid_body_id = m.body_name2id(self.hinged["id"])
        self.knob_geom_id = m.geom_name2id("lid_handle_bar")
        self.hinge_qpos_adr = m.jnt_qposadr[m.joint_name2id(self.joint_name)]
        self.hinge_dof_adr = m.jnt_dofadr[m.joint_name2id(self.joint_name)]
        # pristine (unjittered) body origins, restored by hard_reset rebuild
        self.box_body_pos0 = np.array(m.body_pos[self.box_body_id]).copy()
        self.lid_body_pos0 = np.array(m.body_pos[self.lid_body_id]).copy()

    def _reset_internal(self):
        super()._reset_internal()
        if not self.deterministic_reset:
            dxy = self.rng.uniform(-self.box_jitter, self.box_jitter, 2)
            self.sim.model.body_pos[self.box_body_id][:2] = \
                self.box_body_pos0[:2] + dxy
            self.sim.model.body_pos[self.lid_body_id][:2] = \
                self.lid_body_pos0[:2] + dxy
        # lid starts fully shut, at rest
        self.sim.data.set_joint_qpos(self.joint_name, np.zeros(1))
        self.sim.data.set_joint_qvel(self.joint_name, np.zeros(1))
        self.sim.forward()

    # ------------------------------------------------------------------
    def lid_angle(self):
        return float(self.sim.data.qpos[self.hinge_qpos_adr])

    def lid_speed(self):
        return float(abs(self.sim.data.qvel[self.hinge_dof_adr]))

    def knob_pos(self):
        return np.array(self.sim.data.geom_xpos[self.knob_geom_id])

    def reward(self, action=None):
        return float(self._check_success())

    def _check_success(self, gripper_closed=False):
        return (self.lid_angle() <= self.lid_angle_max and
                self.lid_speed() <= self.lid_speed_max)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset-test", type=int, default=0)
    ap.add_argument("--snap", default=None)
    args = ap.parse_args()

    env = OpenBoxLid(has_renderer=False, use_camera_obs=bool(args.snap))
    obs = env.reset()
    print("[smoke] reset ok; action_dim=%d horizon=%d joint=%s"
          % (env.action_dim, env.horizon, env.joint_name))

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
        box_lo = np.full(2, np.inf)
        box_hi = -np.full(2, np.inf)
        init_success_ok = True
        lid_closed_ok = True
        for i in range(args.reset_test):
            env.reset()
            lid_body = env.sim.data.body_xpos[env.lid_body_id]
            box_lo = np.minimum(box_lo, lid_body[:2])
            box_hi = np.maximum(box_hi, lid_body[:2])
            if env._check_success():
                print("!! reset #%d already succeeds -- bad init" % i)
                init_success_ok = False
            if abs(env.lid_angle()) > 1e-3 or env.lid_speed() > 1e-3:
                print("!! reset #%d lid not shut/at rest: theta=%.4f v=%.4f"
                      % (i, env.lid_angle(), env.lid_speed()))
                lid_closed_ok = False
        span = box_hi - box_lo
        print("[stats] lid hinge world xy min=%s max=%s span=%s"
              % (np.round(box_lo, 4).tolist(), np.round(box_hi, 4).tolist(),
                 np.round(span, 4).tolist()))
        print("[%s] success stays False at init"
              % ("ok" if init_success_ok else "FAIL"))
        print("[%s] lid shut and at rest at every reset"
              % ("ok" if lid_closed_ok else "FAIL"))
        print("[%s] jitter span within +-2*jitter (+1mm tol): span=%s vs %s"
              % ("ok" if np.all(span <= 2 * env.box_jitter + 1e-3) else "FAIL",
                 np.round(span, 4).tolist(), 2 * env.box_jitter))

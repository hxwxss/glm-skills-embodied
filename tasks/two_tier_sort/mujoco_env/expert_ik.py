# -*- coding: utf-8 -*-
"""
expert_ik.py -- M3: IK expert for the two-tier sort task
=========================================================

Phase plan per episode (all world-space keypoints computed from LIVE object
poses, fixed top-down grasp orientation throughout):

  1. LID    : grasp the lid bar (top-down, pads on +-x flanks), follow the
              exact hinge arc past vertical (-110 deg) where gravity holds
              the lid open, release, retreat.
  2. DRAWER : grasp the drawer bar, pull straight out along the slide axis
              (-y) to the joint's open stop, release, retreat.  The slide is
              horizontal, so the drawer stays out with zero actuation.
  3. RED    : pick the red cube off the table, carry at safe height, lower
              into the upper compartment, release.
  4. BLUE   : pick the blue cube, lower into the open drawer tray, release.

  keypoint -> mink numerical IK re-anchored on the live state per keypoint
           -> robosuite JOINT_POSITION controller, absolute joint targets,
              capped per-step deltas
  gripper command lives in its own action dimension (index 7).

Modes:
    python expert_ik.py calib                 # cube grasp-height sweep
    python expert_ik.py test --episodes 6     # acceptance (>= 80%)
    python expert_ik.py demo --out demo.mp4   # record one episode
"""

import argparse
import math
import os
import sys

import numpy as np
import mujoco
import mink

import mjrender as R
from task_two_tier import TwoTierSort

HERE = os.path.dirname(os.path.abspath(__file__))

DQ_MAX = 0.18          # max joint delta per control step (rad): gentle
                       # accelerations keep friction grasps alive
WP_STEP_CAP = 400      # per-waypoint control-step budget
TOL = 6e-3             # IK position tolerance (m)
# palm facing down, Panda fingers separate along the eef x-axis, which this
# (reference-proven) orientation maps to world +-x: perpendicular to both
# the hinge axis and the slide axis, and correct for top-down cube grasps.
GRASP_ROT = np.diag([1.0, -1.0, -1.0])


def joint_controller_config():
    """BASIC composite config, arm segment swapped for absolute JOINT_POSITION."""
    from robosuite import load_composite_controller_config
    from robosuite.controllers import load_part_controller_config
    cfg = load_composite_controller_config(controller="BASIC", robot="Panda")
    part = load_part_controller_config(default_controller="JOINT_POSITION")
    part["input_type"] = "absolute"
    part["interpolation"] = None
    # default kp=50 sags ~2 cm under gravity at tabletop poses; stiffen so
    # joint targets are actually reached (limits raised accordingly)
    part["kp"] = 300
    part["kp_limits"] = [0, 1000]
    right = cfg["body_parts"]["right"]
    gripper_cfg = right.get("gripper", {"type": "GRIP"})
    right.clear()
    right.update(part)
    right["gripper"] = gripper_cfg
    # joint targets exceed +-1 rad: without this the env silently clamps
    right["input_max"] = [4.5] * 7
    right["input_min"] = [-4.5] * 7
    return cfg


def make_env(camera_obs=False, height=256, horizon=4600):
    env = TwoTierSort(
        has_renderer=False,
        has_offscreen_renderer=camera_obs,
        use_camera_obs=camera_obs,
        controller_configs=joint_controller_config(),
        base_types=["RethinkMount"],
        camera_names=["agentview", "sideview"],
        camera_heights=height,
        camera_widths=height,
        horizon=horizon,
    )
    assert env.action_dim == 8, \
        "expected 8 actions (7 arm + 1 gripper), got %d" % env.action_dim
    return env


class IKExpert:
    def __init__(self, env):
        self.rebind(env)

    # ------------------------------------------------------------------
    def rebind(self, env):
        """hard_reset rebuilds MjModel every reset -> rebind everything."""
        self.env = env
        self.m = env.sim.model._model
        self.d = env.sim.data._data
        self.arm_ids = [j for j in range(self.m.njnt)
                        if (mujoco.mj_id2name(self.m,
                            mujoco.mjtObj.mjOBJ_JOINT, j)
                            or "").startswith("robot0_joint")]
        self.arm_adrs = np.array([self.m.jnt_qposadr[j] for j in self.arm_ids])
        self.eef_bid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY,
                                         "gripper0_right_eef")
        self.cfg = mink.Configuration(model=self.m)
        self.task = mink.FrameTask(
            frame_name="gripper0_right_eef", frame_type="body",
            position_cost=10.0, orientation_cost=0.3, lm_damping=1.0)
        self.grasp_rot = mink.SO3.from_matrix(GRASP_ROT)
        arm_save = self.d.qpos[self.arm_adrs].copy()
        mujoco.mj_forward(self.m, self.d)
        self.q_home_arm = arm_save

    def sync(self):
        self.cfg.data.qpos[:] = self.d.qpos
        self.cfg.data.time = self.d.time
        mujoco.mj_forward(self.m, self.cfg.data)

    def eef_pos(self):
        return np.array(self.env.sim.data.xpos[self.eef_bid])

    def solve_ik(self, pos, max_iters=600, tol=3e-3, rot_tol=0.06, seed=None,
                 rot=None, seeded_only=False):
        """Solve to the 6D pose.  `seed` (a previous joint solution) keeps the
        wrist in the same branch between nearby waypoints.  `rot` overrides
        the grasp orientation (used to align the pads with a yawed cube).
        `seeded_only` forbids the unseeded fallbacks: while carrying a cube,
        a wrist-branch switch sweeps the joints wildly and flings the grasp.
        Falls back to live state, then reset home (when allowed).
        Returns (q_arm, pos_err_m, rot_err)."""
        return self._solve_ik_rot(pos, max_iters * (2 if seeded_only else 1),
                                  tol, rot_tol, seed, rot, seeded_only)

    def _solve_ik_rot(self, pos, max_iters, tol, rot_tol, seed, rot,
                      seeded_only=False):
        so3 = (mink.SO3.from_matrix(np.asarray(rot, dtype=float))
               if rot is not None else self.grasp_rot)
        T = mink.SE3.from_rotation_and_translation(
            so3, np.asarray(pos, dtype=float))
        self.task.set_target(T)

        def errors():
            e = np.asarray(self.task.compute_error(self.cfg))
            return float(np.linalg.norm(e[:3])), float(np.linalg.norm(e[3:]))

        def start_from(q_arm_seed):
            self.cfg.data.qpos[:] = self.d.qpos
            if q_arm_seed is not None:
                self.cfg.data.qpos[self.arm_adrs] = q_arm_seed
            self.cfg.data.qvel[:] = 0
            mujoco.mj_forward(self.m, self.cfg.data)

        def iterate():
            perr, rerr = errors()
            for _ in range(max_iters):
                dv = mink.solve_ik(self.cfg, [self.task], dt=1 / 60,
                                   solver="quadprog")
                self.cfg.integrate_inplace(dv, 1 / 60)
                perr, rerr = errors()
                if perr < tol and rerr < rot_tol:
                    break
            return perr, rerr

        if seeded_only and seed is not None:
            start_from(seed)
            perr, rerr = iterate()
            return (self.cfg.data.qpos[self.arm_adrs].copy(), perr, rerr)

        best = None
        for attempt_seed in (seed, None, self.q_home_arm):
            start_from(attempt_seed)
            perr, rerr = iterate()
            if best is None or perr + rerr < best[1] + best[2]:
                best = (self.cfg.data.qpos[self.arm_adrs].copy(),
                        perr, rerr)
            if perr < tol and rerr < rot_tol:
                break
        return best

    # ------------------------------------------------------------------
    def lid_bar_at(self, theta):
        """Lid bar world position when the lid is rotated by theta."""
        r = np.array(self.env.lid_spec["grasp_local"], dtype=float)
        hinge = self.env.hinge_world()
        c, s = math.cos(theta), math.sin(theta)
        return hinge + np.array([r[0], r[1] * c - r[2] * s,
                                 r[1] * s + r[2] * c])

    def _wp(self, pos, gripper, note, dwell=0, rot=None, tol=None):
        wp = dict(pos=np.asarray(pos, dtype=float), gripper=gripper,
                  note=note, dwell=dwell)
        if rot is not None:
            wp["rot"] = rot
        if tol is not None:
            wp["tol"] = tol
        return wp

    def _wp_lazy(self, pos_fn, gripper, note, dwell=0, rot_fn=None,
                 seeded=False):
        return dict(pos=pos_fn, gripper=gripper, note=note, dwell=dwell,
                    rot_fn=rot_fn, seeded=seeded)

    @staticmethod
    def resolve(wp):
        """Resolve a lazy waypoint's position/orientation from the live state."""
        if callable(wp["pos"]):
            wp = dict(wp, pos=np.asarray(wp["pos"](), dtype=float))
        if wp.get("rot_fn") is not None:
            wp = dict(wp, rot=wp["rot_fn"]())
        return wp

    @staticmethod
    def cube_grasp_rot(env, cube):
        """Top-down grasp rotation aligned with the cube's live yaw: the pads
        always square up to the cube faces no matter the spawn yaw."""
        bid = env.obj_body_id[cube]
        quat = env.sim.data.body(bid).xquat           # (w, x, y, z)
        w, x, y, z = quat
        yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        c, s = math.cos(yaw), math.sin(yaw)
        return (np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
                @ GRASP_ROT)

    def _cmd(self, gripper, dwell):
        return dict(pos=None, gripper=gripper, note="cmd", dwell=dwell)

    # ------------------------------------------------------------------
    def lid_phase(self):
        ex_l = self.env.spec["task"]["expert"]["lid"]
        dz = float(ex_l["grasp_dz_m"])
        hover_h = float(ex_l["hover_height_m"])
        bar0 = self.env.lid_bar_pos()
        wps = [self._wp(bar0 + [0, 0, hover_h], -1, "lid-hover"),
               self._wp(bar0 + [0, 0, dz], -1, "lid-descend"),
               self._cmd(1, int(ex_l["close_dwell"])),
               self._wp(bar0 + [0, 0, dz + 0.012], 1, "lid-prelift")]
        thetas = np.linspace(float(ex_l["arc_start_rad"]),
                             float(ex_l["arc_end_rad"]), int(ex_l["arc_steps"]))
        for th in thetas:
            wps.append(self._wp(self.lid_bar_at(th) + [0, 0, dz], 1,
                                "lid-arc-%.2f" % th, int(ex_l["arc_dwell"])))
        nudge = self.lid_bar_at(thetas[-1]) + [0, 0,
                                               dz + float(ex_l["prerelease_lift_m"])]
        wps.append(self._wp(nudge, 1, "lid-prerelease-nudge"))
        wps.append(self._cmd(1, int(ex_l["hold_dwell"])))
        wps.append(self._cmd(-1, int(ex_l["release_dwell"])))
        wps.append(self._wp(nudge + [0, 0, 0.13], -1, "lid-retreat-up"))
        wps.append(self._wp(nudge + [0, 0, 0.13]
                            + np.array(ex_l["retreat_offset_m"]), -1,
                            "lid-retreat-back"))
        return wps

    def drawer_phase(self):
        ex_d = self.env.spec["task"]["expert"]["drawer"]
        dz = float(ex_d["grasp_dz_m"])
        hover_h = float(ex_d["hover_height_m"])
        bar0 = self.env.drawer_bar_pos()
        wps = [self._wp(bar0 + [0, 0, hover_h], -1, "drawer-hover"),
               self._wp(bar0 + [0, 0, dz], -1, "drawer-descend"),
               self._cmd(1, int(ex_d["close_dwell"]))]
        for s in np.linspace(0.0, float(ex_d["pull_distance_m"]),
                             int(ex_d["pull_steps"]) + 1)[1:]:
            wps.append(self._wp(bar0 + [0, -s, dz], 1, "drawer-pull-%.3f" % s,
                                int(ex_d["pull_dwell"])))
        wps.append(self._cmd(1, int(ex_d["release_dwell"])))
        wps.append(self._cmd(-1, int(ex_d["release_dwell"])))
        wps.append(self._wp(bar0 + [0, -float(ex_d["pull_distance_m"]),
                                    dz + float(ex_d["retreat_up_m"])], -1,
                            "drawer-retreat-up"))
        wps.append(self._wp(bar0 + [0, -float(ex_d["pull_distance_m"]), dz]
                            + np.array(ex_d["retreat_offset_m"]), -1,
                            "drawer-retreat-back"))
        return wps

    def pick_phase(self, cube, zone_fn):
        """Pick a cube and place it in its zone.  ALL positions are LAZY
        (callables resolved at execution time): the tray zone only exists
        once the drawer has been pulled, and a gripped cube moves with the
        hand -- plan-time poses are stale.  The grasp orientation aligns
        with the cube's live yaw (captured at rest, first resolution) so
        the pads square up to its faces; it is then FROZEN for the phase:
        a carried cube's quat pendulums under the pads and tracking it
        makes the wrist yaw oscillate until the cube swings out of the
        grip.  Carry waypoints are IK-seeded-only: a wrist-branch switch
        mid-carry sweeps the joints and flings the grasp."""
        ex_p = self.env.spec["task"]["expert"]["pick"]
        dz = float(ex_p["grasp_dz_m"])
        hover_h = float(ex_p["hover_height_m"])
        carry_z = float(ex_p["carry_z_m"])   # absolute world z
        yaw_cell = []

        def rot_fn(c=cube):
            if not yaw_cell:
                yaw_cell.append(self.cube_grasp_rot(self.env, c))
            return yaw_cell[0]

        wps = [
            self._wp_lazy(
                lambda c=cube: self.env.prop_pos(c) + [0, 0, hover_h],
                -1, "%s-hover" % cube, rot_fn=rot_fn),
            self._wp_lazy(
                lambda c=cube: self.env.prop_pos(c) + [0, 0, dz],
                -1, "%s-descend" % cube, rot_fn=rot_fn),
            self._cmd(1, int(ex_p["close_dwell"])),
            self._wp_lazy(
                lambda c=cube: [self.env.prop_pos(c)[0],
                                self.env.prop_pos(c)[1], carry_z],
                1, "%s-lift" % cube, rot_fn=rot_fn, seeded=True),
            self._wp_lazy(
                lambda zf=zone_fn: list(zf()[0][:2]) + [carry_z],
                1, "%s-transit" % cube, rot_fn=rot_fn, seeded=True),
            self._wp_lazy(
                lambda zf=zone_fn, c=cube, dz=dz:
                    list(zf()[0][:2]) + [
                        zf()[0][2]
                        - self.env.prop_specs[c]["dims"][0] / 2
                        + float(ex_p["release_drop_m"]) + dz],
                1, "%s-lower" % cube, rot_fn=rot_fn, seeded=True),
            self._cmd(-1, int(ex_p["release_dwell"])),
            self._cmd(-1, 35),
            self._wp_lazy(
                lambda zf=zone_fn:
                    list(zf()[0][:2])
                    + [zf()[0][2] + float(ex_p["retreat_up_m"])],
                -1, "%s-retreat" % cube),
        ]
        return wps

    def close_drawer_phase(self):
        """Grasp the drawer bar at its pulled-out position and push the
        drawer back to the closed stop."""
        ex_d = self.env.spec["task"]["expert"]["close_drawer"]
        dz = float(ex_d["grasp_dz_m"])
        hover_h = float(ex_d["hover_height_m"])
        s_full = float(self.env.spec["task"]["expert"]["drawer"]
                       ["pull_distance_m"])
        overshoot = float(self.env.spec["task"]["expert"]["close_drawer"]
                          .get("push_overshoot_m", 0.0))
        bar_open = self.env.drawer_bar_pos()   # live: at the open stop
        wps = [self._wp(bar_open + [0, 0, hover_h], -1, "dc-hover"),
               self._wp(bar_open + [0, 0, dz], -1, "dc-descend"),
               self._cmd(1, int(ex_d["close_dwell"]))]
        for s in np.linspace(s_full, -overshoot,
                             int(ex_d["push_steps"]) + 1)[1:]:
            # bar at slide s sits +y of the open position by (s_full - s);
            # s may go slightly negative (past the closed stop) -- the
            # joint limit seats the tray flush
            wps.append(self._wp(bar_open + [0, s_full - s, dz], 1,
                                "dc-push-%.3f" % s, int(ex_d["push_dwell"])))
        wps.append(self._cmd(1, int(ex_d["release_dwell"])))
        wps.append(self._cmd(-1, int(ex_d["release_dwell"])))
        wps.append(self._wp(bar_open + [0, s_full,
                                        dz + float(ex_d["retreat_up_m"])], -1,
                            "dc-retreat-up"))
        wps.append(self._wp(bar_open + [0, s_full, dz]
                            + np.array(ex_d["retreat_offset_m"]), -1,
                            "dc-retreat-back"))
        return wps

    def close_lid_phase(self):
        """Re-grasp the lid bar at its open position, pull the lid forward
        along the hinge arc to just short of shut, and release -- gravity
        closes the last ~9 deg gently.

        At the open stop the bar sits ~0.70 m out and ~1.03 m up, where an
        exact top-down wrist is unreachable (16-19 deg residual that
        wedged the cold-start grasp).  The achievable wrist correction is
        MEASURED once (solve top-down, extract the residual rotation) and
        applied with a linear taper to zero at the closed end."""
        ex_c = self.env.spec["task"]["expert"]["close_lid"]
        ex_l = self.env.spec["task"]["expert"]["lid"]
        dz = float(ex_c["grasp_dz_m"])
        hover_h = float(ex_c["hover_height_m"])
        th_end = float(ex_l["arc_end_rad"])
        th_close = float(ex_c["arc_end_rad"])
        # grasp the bar WHERE IT IS: after a partial close attempt the lid
        # rests somewhere mid-arc and the arc-end position would grasp air
        bar_open = self.env.lid_bar_pos()
        th_start = float(np.clip(self.env.lid_angle(),
                                 min(th_close, th_end), max(th_close, th_end)))

        # approach HIGH and over the top: a direct path from the front
        # cuts through the standing lid panel (presses it) while the
        # swinging elbow hooks the closed drawer handle and yanks it open
        # (observed: drawer yanked 0 -> 0.08 m in one descend)
        via1 = np.array([bar_open[0], -0.10, 1.30])
        via2 = np.array([bar_open[0], bar_open[1], 1.30])

        # measure the achievable wrist correction at the grasp pose
        q_arm, perr, rerr = self.solve_ik(bar_open + [0, 0, dz])
        self.cfg.data.qpos[:] = self.d.qpos
        self.cfg.data.qpos[self.arm_adrs] = q_arm
        mujoco.mj_forward(self.m, self.cfg.data)
        R_ach = np.array(self.cfg.data.xmat[self.eef_bid]).reshape(3, 3)
        R_err = R_ach @ GRASP_ROT.T
        w = np.array([R_err[2, 1] - R_err[1, 2],
                      R_err[0, 2] - R_err[2, 0],
                      R_err[1, 0] - R_err[0, 1]])
        max_ang = float(np.arctan2(np.linalg.norm(w),
                                   np.trace(R_err) - 1.0))
        axis = w / (np.linalg.norm(w) + 1e-12)
        K = np.array([[0.0, -axis[2], axis[1]],
                      [axis[2], 0.0, -axis[0]],
                      [-axis[1], axis[0], 0.0]])

        def cl_rot(th):
            """Measured correction, tapered to zero at the closed end."""
            t = max(0.0, (th - th_close) / max(th_start - th_close, 1e-6))
            ang = max_ang * t
            c, s = math.cos(ang), math.sin(ang)
            return (np.eye(3) + s * K + (1.0 - c) * (K @ K)) @ GRASP_ROT

        wps = [self._wp(via1, -1, "cl-via1", rot=cl_rot(th_start)),
               self._wp(via2, -1, "cl-via2", rot=cl_rot(th_start)),
               self._wp(bar_open + [0, 0, hover_h], -1, "cl-hover",
                        rot=cl_rot(th_start)),
               self._wp(bar_open + [0, 0, dz], -1, "cl-descend",
                        rot=cl_rot(th_start)),
               self._cmd(1, int(ex_c["close_dwell"])),
               self._wp(bar_open + [0, 0, dz + 0.008], 1, "cl-seat",
                        rot=cl_rot(th_start))]
        for th in np.linspace(th_start, th_close, int(ex_c["arc_steps"])):
            wps.append(self._wp(self.lid_bar_at(th) + [0, 0, dz], 1,
                                "cl-arc-%.2f" % th, int(ex_c["arc_dwell"]),
                                rot=cl_rot(th), tol=0.06))
        # release WHILE retreating up+back: a frozen release lets the
        # falling lid's handle land in the open jaw, which cradles the lid
        # half-open (observed: lid stuck at -0.42 rad)
        wps.append(self._wp(self.lid_bar_at(th_close) + [0, -0.05, dz + 0.11],
                            -1, "cl-release", int(ex_c["release_dwell"])))
        wps.append(self._wp(self.lid_bar_at(th_close)
                            + [0, -0.05 - 0.04, dz + 0.11 + 0.05], -1,
                            "cl-retreat-up"))
        wps.append(self._wp(self.lid_bar_at(th_close)
                            + [0, -0.05 - 0.08, dz + 0.11 + 0.05]
                            + np.array(ex_c["retreat_offset_m"]) * 0.5, -1,
                            "cl-retreat-back"))
        return wps

    def final_phase(self):
        settle = int(self.env.spec["task"]["expert"]["settle_dwell"])
        # exit pose: stay HIGH and slide right-front -- a low exit path
        # sweeps past the closed lid handle and the closed drawer handle
        # and hooks them back open (both observed in acceptance runs)
        cur = np.array(self.env.sim.data.xpos[self.eef_bid])
        wps = [self._cmd(-1, settle),
               self._wp(cur + np.array([0.12, -0.22, 0.06]), -1,
                        "final-retreat")]
        return wps

    # ------------------------------------------------------------------
    def build_phases(self):
        """Ordered (name, builder, verifier) phases.  Verifiers gate the
        retry loop: a phase whose outcome does not hold is re-attempted
        from the live state (lazy waypoints re-resolve)."""

        def lid_ok():
            return self.env.lid_angle() <= -1.62

        def drawer_ok():
            return self.env.drawer_slide() >= 0.08

        def cube_ok(cube, zone_fn):
            def verify():
                pos = self.env.prop_pos(cube)
                center, half = zone_fn()
                # +2 mm slack: a cube resting flush against a tray wall
                # sits exactly on the boundary
                if not np.all(np.abs(pos - center) < half + 0.002):
                    return False
                return self.env.prop_speed(cube) < 0.06
            return verify

        def lid_closed():
            return abs(self.env.lid_angle()) <= float(
                self.env.spec["task"]["success_condition"]["lid_closed_rad"])

        def drawer_closed():
            return self.env.drawer_slide() <= float(
                self.env.spec["task"]["success_condition"]["drawer_closed_m"])

        return [
            ("open lid", self.lid_phase, lid_ok),
            ("pull drawer", self.drawer_phase, drawer_ok),
            ("place RedCube", lambda: self.pick_phase(
                "RedCube", self.env.upper_zone_world),
             cube_ok("RedCube", self.env.upper_zone_world)),
            ("place BlueCube", lambda: self.pick_phase(
                "BlueCube", self.env.tray_zone_world),
             cube_ok("BlueCube", self.env.tray_zone_world)),
            ("close drawer", self.close_drawer_phase, drawer_closed),
            ("close lid", self.close_lid_phase, lid_closed),
            ("final settle", self.final_phase, None),
        ]

    def cube_placed(self, cube, zone_fn):
        pos = self.env.prop_pos(cube)
        center, half = zone_fn()
        return bool(np.all(np.abs(pos - center) < half + 0.002))

    def plan(self):
        """Flattened waypoint list (reachability pre-check / diagnostics)."""
        flat = []
        for _, builder, _ in self.build_phases():
            flat.extend(builder())
        return flat

    def gripper_spread(self):
        return abs(float(self.d.qpos[9]) - float(self.d.qpos[10])) * 2000.0

    def ensure_open(self, env, min_spread_m=0.130):
        """Pop a jammed finger: if the gripper is not fully open, pulse
        close-then-open so both pads move together and release the wedge."""
        for _ in range(2):
            if self.gripper_spread() >= min_spread_m:
                return True
            for gripper, dwell in ((1.0, 12), (-1.0, 50)):
                hold = np.zeros(env.action_dim)
                hold[:7] = self.d.qpos[self.arm_adrs]
                hold[7] = gripper
                for _ in range(dwell):
                    env.step(hold)
        return self.gripper_spread() >= min_spread_m

    # ------------------------------------------------------------------
    def _track(self, env, q_arm, gripper, record_hook=None, tol=0.03,
               settle_steps=15, cap=WP_STEP_CAP):
        """Step joint targets toward q_arm; done only after the controller
        has HELD within tol for `settle_steps` consecutive steps."""
        held = 0
        jsteps = 0
        for _ in range(cap):
            qa = self.d.qpos[self.arm_adrs]
            a = np.zeros(env.action_dim)
            a[:7] = qa + np.clip(q_arm - qa, -DQ_MAX, DQ_MAX)
            a[7] = float(gripper)
            obs, _, done, _ = env.step(a)
            jsteps += 1
            if record_hook is not None:
                record_hook(jsteps, obs)
            if np.all(np.abs(q_arm - qa) < tol):
                held += 1
                if held >= settle_steps:
                    return done, jsteps
            else:
                held = 0
            if done:
                return True, jsteps
        return False, jsteps

    def recovery_lift(self, env, cube):
        """After a failed place attempt, climb back to carry height above
        the cube's live position and let the scene settle before the retry
        descent (a blind re-descend can press the hand into the cube)."""
        carry_z = float(self.env.spec["task"]["expert"]["pick"]["carry_z_m"])
        pos = self.env.prop_pos(cube)
        q_arm, _, _ = self.solve_ik([pos[0], pos[1], carry_z])
        self._track(env, q_arm, -1.0)
        hold = np.zeros(env.action_dim)
        hold[:7] = self.d.qpos[self.arm_adrs]
        hold[7] = -1.0
        for _ in range(30):
            env.step(hold)

    def execute(self, env, phases=None, record_hook=None, step_cb=None,
                verbose=False, max_attempts=3):
        """Run every phase; each phase is retried (up to `max_attempts`)
        until its verifier holds.  step_cb(action, obs) fires after every
        env.step (dataset recording / contact audits)."""
        self.rebind(env)
        if phases is None:
            phases = self.build_phases()
        jsteps = 0
        prev_q = None
        info = {}
        done = False
        for name, builder, verify in phases:
            ok = False
            for attempt in range(1, max_attempts + 1):
                if name.startswith(("place", "close")):
                    self.ensure_open(env)
                    if attempt > 1:
                        cube = "RedCube" if "Red" in name else "BlueCube"
                        zone_fn = (self.env.upper_zone_world
                                   if cube == "RedCube"
                                   else self.env.tray_zone_world)
                        if not self.cube_placed(cube, zone_fn):
                            self.recovery_lift(env, cube)
                if verbose and attempt > 1:
                    print("   [retry %d] %s" % (attempt, name))
                for wp in builder():
                    if wp["pos"] is not None:
                        wp = self.resolve(wp)
                        q_arm, perr, rerr = self.solve_ik(
                            wp["pos"], seed=prev_q, rot=wp.get("rot"),
                            seeded_only=wp.get("seeded", False))
                        prev_q = q_arm
                        if verbose and (perr > 0.01 or rerr > 0.08):
                            print("   [ik] %s pos %.3f rot %.3f"
                                  % (wp["note"], perr, rerr))
                        done, n = self._track(env, q_arm, wp["gripper"],
                                              record_hook=record_hook,
                                              tol=wp.get("tol", 0.03))
                        jsteps += n
                        if verbose:
                            print("   [wp] %-22s steps=%d"
                                  % (wp["note"], n))
                        dwell = int(wp.get("dwell") or 0)
                        if dwell and not done:
                            hold = np.zeros(env.action_dim)
                            hold[:7] = self.d.qpos[self.arm_adrs]
                            hold[7] = float(wp["gripper"])
                            for _ in range(dwell):
                                obs, _, done, _ = env.step(hold)
                                jsteps += 1
                                if step_cb is not None:
                                    step_cb(hold, obs)
                                if record_hook is not None:
                                    record_hook(jsteps, obs)
                                if done:
                                    break
                    else:
                        hold = np.zeros(env.action_dim)
                        hold[:7] = self.d.qpos[self.arm_adrs]
                        hold[7] = float(wp["gripper"])
                        for _ in range(int(wp.get("dwell") or 30)):
                            obs, _, done, _ = env.step(hold)
                            jsteps += 1
                            if step_cb is not None:
                                step_cb(hold, obs)
                            if record_hook is not None:
                                record_hook(jsteps, obs)
                            if done:
                                break
                    if done:
                        break
                if done:
                    break
                if verify is None or verify():
                    ok = True
                    break
                if verbose:
                    print("   [verify FAILED] %s (attempt %d)"
                          % (name, attempt))
            info[name] = ok
            if done:
                break
        info["lid"] = env.lid_angle()
        info["drawer"] = env.drawer_slide()
        return env._check_success(), jsteps, info

    def run(self, env, record_hook=None, verbose=False):
        return self.execute(env, record_hook=record_hook, verbose=verbose)

# ---------------------------------------------------------------------------
def calib(dz_values):
    """Cube grasp-height sweep: descend + close at each dz, report contacts
    and fingertip offsets relative to the cube center."""
    env = make_env()
    ex = IKExpert(env)
    m = env.sim.model._model

    cube_gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "RedCube_g0")
    if cube_gid < 0:
        cube_gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM,
                                     "RedCube_geom")
    table_gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM,
                                  "table_collision")

    def contacts_for(gid):
        names = set()
        for i in range(env.sim.data._data.ncon):
            c = env.sim.data._data.contact[i]
            if c.dist >= 0:
                continue
            for g in (c.geom1, c.geom2):
                if g == gid:
                    other = c.geom2 if g == c.geom1 else c.geom1
                    names.add(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM,
                                                other) or "?")
        return names

    finger_geoms = [g for g in range(m.ngeom)
                    if "finger" in (mujoco.mj_id2name(m,
                        mujoco.mjtObj.mjOBJ_GEOM, g) or "")
                    and "collision" in (mujoco.mj_id2name(m,
                        mujoco.mjtObj.mjOBJ_GEOM, g) or "")]

    print("cube grasp dz sweep (fingers must touch cube, never table):")
    for dz in dz_values:
        env.reset()
        ex.rebind(env)
        ex_p = env.spec["task"]["expert"]["pick"]
        hover_h = float(ex_p["hover_height_m"])
        c0 = env.prop_pos("RedCube")
        # hover -> descend -> close only
        for pos, grip, dwell in ((c0 + [0, 0, hover_h], -1, 0),
                                 (c0 + [0, 0, dz], -1, 0), (None, 1, 45)):
            if pos is not None:
                q_arm, perr, _ = ex.solve_ik(pos)
                if perr > 0.02:
                    print("  dz=%+.3f IK failed (perr=%.1f mm)"
                          % (dz, perr * 1e3))
                    break
                ex._track(env, q_arm, grip)
            else:
                hold = np.zeros(env.action_dim)
                hold[:7] = ex.d.qpos[ex.arm_adrs]
                hold[7] = 1.0
                for _ in range(dwell):
                    env.step(hold)
        cube_touch = contacts_for(cube_gid)
        table_touch = contacts_for(table_gid)
        mujoco.mj_forward(ex.m, ex.d)
        fps = [np.array(ex.d.geom_xpos[g]) for g in finger_geoms]
        mid = (fps[0] + fps[1]) / 2 if fps else np.zeros(3)
        cnow = env.prop_pos("RedCube")
        print("  dz=%+.3f  finger->cube: %s  touch_table: %s  "
              "mid-cube=(%s) mm  cube_dz=%+.1f mm"
              % (dz, sorted(n for n in cube_touch if "finger" in n) or "none",
                 sorted(n for n in table_touch if "finger" in n) or "none",
                 np.round((mid - cnow) * 1000, 0),
                 1000 * float(mid[2] - cnow[2]) if fps else 0))
    env.close()


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["calib", "test", "demo"])
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--dz", type=float, nargs="*",
                    default=[-0.005, 0.0, 0.005, 0.01, 0.02])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.mode == "calib":
        calib(args.dz)
        sys.exit(0)

    import imageio
    env = make_env()
    ex = IKExpert(env)
    ren = R.make_renderer(env, height=480, width=640)
    cams = {n: R.camera_fixed(env, n) for n in ("agentview", "sideview")}

    def snap():
        a = R.render_snapshot(ren, env, cams["agentview"])
        s = R.render_snapshot(ren, env, cams["sideview"])
        return np.concatenate([a, s], axis=1)

    wins = 0
    saved = 0
    for ep in range(args.episodes):
        env.reset()
        ex.rebind(env)
        for name, cam in cams.items():
            cam.fixedcamid = mujoco.mj_name2id(
                env.sim.model._model, mujoco.mjtObj.mjOBJ_CAMERA, name)
        frames = []
        hook = (lambda j, o, _f=frames: _f.append(snap())) \
            if (args.mode == "demo" and saved == 0) else None
        ok, steps, info = ex.run(env, record_hook=hook, verbose=True)
        wins += int(ok)
        print("[episode %d] success=%s steps=%d lid=%.2f drawer=%.3f info=%s"
              % (ep, ok, steps, env.lid_angle(), env.drawer_slide(), info))
        if ok and args.mode == "demo" and saved == 0 and frames:
            out = args.out or os.path.join(HERE, "..", "rollouts",
                                           "expert_demo.mp4")
            os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
            imageio.mimsave(out, frames, fps=int(env.control_freq))
            print("demo video ->", out)
            saved += 1
    rate = wins / max(args.episodes, 1)
    print("SUCCESS_RATE=%d/%d = %.0f%%" % (wins, args.episodes, 100 * rate))
    ren.close()
    env.close()
    sys.exit(0 if rate >= 0.8 else 1)


if __name__ == "__main__":
    main()

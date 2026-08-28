# -*- coding: utf-8 -*-
"""
expert_ik.py -- M3: IK expert for the lid-opening task
=======================================================

Strategy: the gripper grasps the knob on the lid's front edge (top-down,
fingers perpendicular to the hinge axis) and follows the knob's exact hinge
arc past vertical, where gravity holds the lid against the open stop.

  world-space keypoints from live object poses
    -> mink numerical IK re-anchored on the live state per keypoint
    -> robosuite JOINT_POSITION controller, absolute joint targets,
       capped per-step deltas
  gripper command lives in its own action dimension (index 7).

Modes:
    python expert_ik.py calib                 # grasp-height calibration sweep
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
from task_open_lid import OpenBoxLid

HERE = os.path.dirname(os.path.abspath(__file__))

DQ_MAX = 0.25          # max joint delta per control step (rad)
WP_STEP_CAP = 400      # per-waypoint control-step budget
TOL = 6e-3             # IK position tolerance (m)
# palm facing down, Panda fingers separate along the eef x-axis, which this
# (reference-proven) orientation maps to world +-x: perpendicular to the
# hinge axis, clear of the lid's tilt plane (verified empirically in calib).
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


def make_env(camera_obs=False, height=256, horizon=900):
    env = OpenBoxLid(
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

    def solve_ik(self, pos, max_iters=600, tol=TOL, rot_tol=0.06, seed=None):
        """Solve to the 6D pose.  `seed` (a previous joint solution) keeps the
        wrist in the same branch between nearby waypoints -- without it the
        solver intermittently flips ~2 rad to the mirrored wrist branch
        mid-arc and rips the gripper open.  Falls back to live state, then
        reset home.  Returns (q_arm, pos_err_m, rot_err_rad)."""
        T = mink.SE3.from_rotation_and_translation(
            self.grasp_rot, np.asarray(pos, dtype=float))
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
    def knob_world_at(self, theta, hinge_pos):
        """Knob-head world position when the lid is rotated by theta."""
        r = self.env.knob_local
        c, s = math.cos(theta), math.sin(theta)
        return hinge_pos + np.array([r[0],
                                     r[1] * c - r[2] * s,
                                     r[1] * s + r[2] * c])

    def plan(self, dz_override=None):
        ex = self.env.spec["task"]["expert"]
        dz = float(dz_override if dz_override is not None
                   else ex["grasp_dz_m"])
        hover_h = float(ex["hover_height_m"])
        th0, th1 = float(ex["arc_start_rad"]), float(ex["arc_end_rad"])
        n_arc = int(ex["arc_steps"])

        head0 = self.env.knob_pos()
        hinge = np.array(self.env.sim.data.body_xpos[self.env.lid_body_id])

        W = [
            dict(pos=head0 + np.array([0, 0, hover_h]), gripper=-1,
                 note="hover-above-knob"),
            dict(pos=head0 + np.array([0, 0, dz]), gripper=-1,
                 note="descend-to-grasp"),
            dict(pos=None, gripper=1, note="close-gripper",
                 dwell=int(ex["close_dwell"])),
            dict(pos=head0 + np.array([0, 0, dz + 0.012]), gripper=1,
                 note="prelift"),
        ]
        for i, theta in enumerate(np.linspace(th0, th1, n_arc)):
            p = self.knob_world_at(theta, hinge) + np.array([0, 0, dz])
            W.append(dict(pos=p, gripper=1, note="arc-%.2f" % theta,
                          dwell=int(ex.get("arc_dwell", 0))))
        arc_end = W[-1]["pos"]
        # lift the hand clear of the raised lid while still holding the knob
        prelift_open = float(ex.get("prerelease_lift_m", 0.05))
        nudge = arc_end + np.array([0, 0, prelift_open])
        W.append(dict(pos=nudge, gripper=1, note="prerelease-nudge"))
        W.append(dict(pos=None, gripper=1, note="hold-open",
                      dwell=int(ex["hold_dwell"])))
        W.append(dict(pos=None, gripper=-1, note="release",
                      dwell=int(ex["release_dwell"])))
        # retreat straight up first, then back toward the robot
        W.append(dict(pos=nudge + np.array([0, 0, 0.13]), gripper=-1,
                      note="retreat-up"))
        W.append(dict(pos=nudge + np.array([0, 0, 0.13])
                      + np.array(ex["retreat_offset_m"]), gripper=-1,
                      note="retreat-back"))
        W.append(dict(pos=None, gripper=-1,
                      dwell=int(ex["settle_dwell"]), note="settle"))
        return W

    # ------------------------------------------------------------------
    def _track(self, env, q_arm, gripper, record_hook=None, tol=0.03,
               settle_steps=15, cap=WP_STEP_CAP):
        """Step joint targets toward q_arm; done only after the controller
        has HELD within tol for `settle_steps` consecutive steps (a single
        in-tolerance sample can occur mid-transit while the controller is
        still settling toward the previous waypoint)."""
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

    def run(self, env, record_hook=None, verbose=False):
        self.rebind(env)
        wps = self.plan()
        jsteps = 0
        opened_seen = False
        prev_q = None
        for wp in wps:
            if wp["pos"] is not None:
                q_arm, perr, rerr = self.solve_ik(wp["pos"], seed=prev_q)
                prev_q = q_arm
                if verbose and (perr > 0.01 or rerr > 0.08):
                    print("   [ik] %s pos %.3f rot %.3f"
                          % (wp["note"], perr, rerr))
                done, n = self._track(env, q_arm, wp["gripper"], record_hook)
                jsteps += n
                # optional settle at the waypoint (arc waypoints use this so
                # the lid catches up before the next one)
                dwell = int(wp.get("dwell") or 0)
                if dwell and not done:
                    hold = np.zeros(env.action_dim)
                    hold[:7] = self.d.qpos[self.arm_adrs]
                    hold[7] = float(wp["gripper"])
                    for _ in range(dwell):
                        obs, _, done, _ = env.step(hold)
                        jsteps += 1
                        if record_hook is not None:
                            record_hook(jsteps, obs)
                        if done:
                            break
                if done:
                    return env._check_success(), jsteps, {"phase": wp["note"]}
            else:
                hold = np.zeros(env.action_dim)
                hold[:7] = self.d.qpos[self.arm_adrs]
                hold[7] = float(wp["gripper"])
                for _ in range(int(wp.get("dwell") or 30)):
                    obs, _, done, _ = env.step(hold)
                    jsteps += 1
                    if record_hook is not None:
                        record_hook(jsteps, obs)
                    if done:
                        return env._check_success(), jsteps, {"phase": wp["note"]}
            if env.lid_angle() <= self.env.lid_angle_max:
                opened_seen = True
        return env._check_success(), jsteps, {"opened_seen": opened_seen}


# ---------------------------------------------------------------------------
def calib(dz_values):
    """Grasp-height sweep: descend + close at each dz, report contacts."""
    env = make_env()
    ex = IKExpert(env)
    m = env.sim.model._model
    head_gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "lid_handle_bar")
    panel_gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "lid_panel")
    finger_geoms = [g for g in range(m.ngeom)
                    if "finger" in (mujoco.mj_id2name(m,
                        mujoco.mjtObj.mjOBJ_GEOM, g) or "")
                    and "collision" in (mujoco.mj_id2name(m,
                        mujoco.mjtObj.mjOBJ_GEOM, g) or "")]

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

    print("dz sweep (fingers must touch head, never lid_panel):")
    for dz in dz_values:
        env.reset()
        ex.rebind(env)
        wps = ex.plan(dz_override=dz)
        ok_move = True
        # execute hover -> descend -> close only
        for wp in wps[:3]:
            if wp["pos"] is not None:
                q_arm, perr, rerr = ex.solve_ik(wp["pos"])
                if perr > 0.02:
                    ok_move = False
                    break
                ex._track(env, q_arm, wp["gripper"])
            else:
                hold = np.zeros(env.action_dim)
                hold[:7] = ex.d.qpos[ex.arm_adrs]
                hold[7] = float(wp["gripper"])
                for _ in range(int(wp.get("dwell") or 20)):
                    env.step(hold)
        head_touch = contacts_for(head_gid)
        panel_touch = contacts_for(panel_gid)
        fng = sorted(n for n in head_touch if "finger" in n)
        lid_fng = sorted(n for n in panel_touch if "finger" in n)
        # finger geometry relative to the knob head
        mujoco.mj_forward(ex.m, ex.d)
        fps = [np.array(ex.d.geom_xpos[g]) for g in finger_geoms]
        mid = (fps[0] + fps[1]) / 2 if fps else np.zeros(3)
        head_now = np.array(ex.d.geom_xpos[head_gid])
        print("  dz=%+.3f  ik_ok=%s  finger->head: %s  touch_lid: %s  "
              "mid-head=(%s) mm  gap=%.1f mm"
              % (dz, ok_move, fng or "none", lid_fng or "none",
                 np.round((mid - head_now) * 1000, 0),
                 1000 * float(np.linalg.norm(fps[0] - fps[1])) if len(fps) == 2 else -1))
    env.close()


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["calib", "test", "demo"])
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--dz", type=float, nargs="*",
                    default=[-0.01, 0.0, 0.01, 0.02, 0.03, 0.045])
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
        print("[episode %d] success=%s steps=%d lid=%.2f info=%s"
              % (ep, ok, steps, env.lid_angle(), info))
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

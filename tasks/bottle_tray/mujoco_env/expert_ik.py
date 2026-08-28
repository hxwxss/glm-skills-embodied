# -*- coding: utf-8 -*-
"""
expert_ik.py — offline IK + joint-position-control expert (M3)
===============================================================

World-space keypoints planned from the LIVE object poses each episode
(bottle pose from placement sampler, tray pose from the body_pos jitter):

    hover above grasp band -> descend -> close -> lift -> above tray
    -> lower (bottle base just above tray floor) -> release -> retreat

Each keypoint is solved with mink numerical IK from the CURRENT live state
(local-minimum retry from the reset home), then executed with the
JOINT_POSITION controller in absolute mode (capped per-step joint delta).
The Panda stands on the floor at the table edge (NullMount from the IR —
no tall pedestal).

Usage:
    python expert_ik.py calib                 # in-sim gripper/palm calibration
    python expert_ik.py test --episodes 8     # success rate (accept >= 80%)
    python expert_ik.py demo --out demo.mp4   # record one review video
"""

import argparse
import math
import os
import sys

import numpy as np
import mujoco
import mink

from task_bottle_in_tray import PutBottleInTray

HERE = os.path.dirname(os.path.abspath(__file__))

# --- tunables (fixed after calibration) -------------------------------
GRASP_BAND_OFF = 0.04    # EE grasp target = bottle_center_z + this
LIFT_H = 0.16            # lift height above bottle center
DROP_CLEAR = 0.006       # bottle base clearance above tray floor at release
TRANSIT_ABOVE_TABLE = 0.16   # transit height (EE) above the table
DQ_MAX = 0.25            # per-step joint delta cap (rad)
DQ_SLOW = 0.08           # gentle cap for the tray descent
ARRIVE_TOL = 0.06        # joint-space arrival tolerance (rad)
SETTLE_DWELL = 8


def joint_controller_config():
    """BASIC composite config, right-arm segment = absolute joint position."""
    from robosuite import load_composite_controller_config
    from robosuite.controllers import load_part_controller_config
    cfg = load_composite_controller_config(controller="BASIC", robot="Panda")
    part = load_part_controller_config(default_controller="JOINT_POSITION")
    part["input_type"] = "absolute"
    part["interpolation"] = None
    right = cfg["body_parts"]["right"]
    gripper_cfg = right.get("gripper", {"type": "GRIP"})
    right.clear()
    right.update(part)
    right["gripper"] = gripper_cfg
    # joint angles exceed +-pi; robosuite default clips inputs to +-1 which
    # silently truncates absolute targets
    right["input_max"] = [4.5] * 7
    right["input_min"] = [-4.5] * 7
    return cfg


def make_env(camera_obs=False, height=256, horizon=1500):
    # uses the IR's mount=floor -> NullMount: Panda stands at the table edge
    return PutBottleInTray(
        has_renderer=False,
        has_offscreen_renderer=camera_obs,
        use_camera_obs=camera_obs,
        controller_configs=joint_controller_config(),
        camera_names=["agentview"],
        camera_heights=height,
        camera_widths=height,
        horizon=horizon,
    )


class IKExpert:
    def __init__(self, env):
        self._bind(env)

    def _bind(self, env):
        self.env = env
        sp = env.spec
        b = next(o for o in sp["objects"] if o["id"] == "GreenBottle")
        self.bottle_h = b["dims"][1]
        self.bottle_r = b["dims"][0]
        self.tray = next(o for o in sp["objects"]
                         if "container" in o.get("semantic", []))
        self.tray_wall_t = self.tray["walls"][0]["size"][2] * 0 + \
            self.tray["walls"][-1]["size"][2]   # bottom thickness
        self.table_top_z = float(sp["workspace"]["table_top_z"])

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
            position_cost=10.0, orientation_cost=0.1, lm_damping=1.0)
        # posture bias: keeps solutions in the elbow-up family (avoids
        # self-collision and most table-edge grazing). Floating-base coords
        # are ignored by the task; the target is sized to nq.
        self.posture = mink.PostureTask(model=self.m, cost=0.5, lm_damping=1.0)
        q_ref = np.array(self.d.qpos, dtype=float)
        home = np.array(env.spec["robots"][0].get("reset_home_joints"),
                        dtype=float)
        q_ref[self.arm_adrs] = home
        self.posture.set_target(q_ref)
        self.tasks = [self.task, self.posture]
        R_t = np.eye(3); R_t[1, 1] = -1; R_t[2, 2] = -1   # fingers down
        self.grasp_rot = mink.SO3.from_matrix(R_t)
        arm_save = self.d.qpos[self.arm_adrs].copy()
        mujoco.mj_forward(self.m, self.d)
        self.q_home_arm = arm_save

    def sync(self):
        self.cfg.data.qpos[:] = self.d.qpos
        self.cfg.data.time = self.d.time
        mujoco.mj_forward(self.m, self.cfg.data)

    rebind = _bind

    def _arm_collisions(self, data=None):
        """Arm-vs-world/arm-vs-arm contact pairs in the given mjData.

        Returns a list of (depth, name1, name2) for contacts involving a
        robot geom, excluding legitimate finger-bottle grasp contacts.
        """
        data = data if data is not None else self.cfg.data
        out = []
        for i in range(data.ncon):
            c = data.contact[i]
            b1 = (mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_BODY,
                                   self.m.geom_bodyid[c.geom1]) or "")
            b2 = (mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_BODY,
                                   self.m.geom_bodyid[c.geom2]) or "")
            arm1 = b1.startswith("robot0") or b1.startswith("gripper0")
            arm2 = b2.startswith("robot0") or b2.startswith("gripper0")
            if not (arm1 or arm2):
                continue
            grasp = (("finger" in b1 or "finger" in b2)
                     and ("GreenBottle" in b1 or "GreenBottle" in b2))
            if grasp:
                continue
            out.append((-c.dist, b1, b2))
        return out

    def solve_ik(self, pos, max_iters=400, tol=6e-3):
        """Solve to `pos`; reject solutions that collide (deep table-edge
        grazing or arm self-collision) and retry from alternate seeds."""
        T = mink.SE3.from_rotation_and_translation(
            self.grasp_rot, np.asarray(pos, dtype=float))
        self.task.set_target(T)

        def iterate():
            err = float("inf")
            for _ in range(max_iters):
                dv = mink.solve_ik(self.cfg, self.tasks, dt=1 / 60,
                                   solver="quadprog")
                self.cfg.integrate_inplace(dv, 1 / 60)
                err = float(np.linalg.norm(self.cfg.data.xpos[self.eef_bid] - pos))
                if err < tol:
                    break
            return err

        seeds = [None, self.q_home_arm]  # live state, then reset home
        rng = np.random.RandomState(0)
        for _ in range(6):
            seeds.append(self.q_home_arm +
                         rng.uniform(-0.4, 0.4, size=self.q_home_arm.shape))
        for attempt, seed in enumerate(seeds):
            if attempt == 0:
                self.sync()
            else:
                self.cfg.data.qpos[:] = self.d.qpos
                self.cfg.data.qpos[self.arm_adrs] = seed
                self.cfg.data.qvel[:] = 0
                mujoco.mj_forward(self.m, self.cfg.data)
            err = iterate()
            coll = self._arm_collisions()
            if err <= tol and not coll:
                q_arm = self.cfg.data.qpos[self.arm_adrs].copy()
                return q_arm, err
        # best-effort: return the last solution (execution will report)
        q_arm = self.cfg.data.qpos[self.arm_adrs].copy()
        return q_arm, err

    def plan(self, bottle_pos, tray_pos):
        bx, by, bz = bottle_pos
        tx, ty, tz = tray_pos
        tray_floor = tz + self.tray["walls"][-1]["size"][2]  # + bottom thickness
        z_grasp = bz + GRASP_BAND_OFF
        z_lip = self.table_top_z + self.tray["walls"][0]["size"][2]  # wall top
        z_release = (tray_floor + DROP_CLEAR + self.bottle_h / 2
                     + GRASP_BAND_OFF)
        z_transit = max(bz + LIFT_H, self.table_top_z + TRANSIT_ABOVE_TABLE)
        W = [
            dict(pos=np.array([bx, by, z_grasp + 0.12]), gripper=-1,
                 note="hover-above-bottle"),
            dict(pos=np.array([bx, by, z_grasp]), gripper=-1,
                 note="descend-to-grasp"),
            dict(pos=None, gripper=1, note="close-gripper", dwell=35),
            dict(pos=np.array([bx, by, bz + LIFT_H]), gripper=1, note="lift"),
            dict(pos=np.array([tx, ty, z_transit]), gripper=1,
                 note="move-above-tray"),
            dict(pos=np.array([tx, ty, z_release + 0.05]), gripper=1,
                 note="align-over-mouth", dwell=5),
            dict(pos=np.array([tx, ty, z_release]), gripper=1,
                 note="lower-into-tray", dwell=10, dq=DQ_SLOW),
            dict(pos=None, gripper=-1, note="release", dwell=30),
            dict(pos=np.array([tx, ty, z_release + 0.12]), gripper=-1,
                 note="retreat"),
        ]
        return W

    def run(self, env, record_hook=None, verbose=False):
        # hard_reset rebuilds the MjModel: re-bind mink config every episode
        self.rebind(env)
        b0 = env.bottle_pos().copy()
        t0 = env.tray_center().copy()
        wps = self.plan(b0, t0)
        jsteps = 0
        done = False
        for wp in wps:
            dq_cap = float(wp.get("dq", DQ_MAX))
            if wp["pos"] is not None:
                q_arm, err = self.solve_ik(wp["pos"])
                if err > 0.02:
                    print(f"   [warn] IK residual {err:.3f} at {wp['note']}")
                for _ in range(400):
                    qa = self.d.qpos[self.arm_adrs]
                    if np.all(np.abs(q_arm - qa) < ARRIVE_TOL):
                        break
                    a = np.zeros(env.action_dim)
                    a[:7] = qa + np.clip(q_arm - qa, -dq_cap, dq_cap)
                    a[7] = float(wp["gripper"])
                    obs, _, done, _ = env.step(a)
                    jsteps += 1
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
                    if record_hook is not None:
                        record_hook(jsteps, obs)
                    if done:
                        break
            if done:
                break
            if verbose:
                print(f"   phase={wp['note']:18s} steps={jsteps}")
        return env._check_success(), jsteps, {"phase": wps[-1]["note"]}


# ---------------------------------------------------------------------------
def _fingertip_stats(expert):
    """Effective palm->fingertip geometry in the current pose (in-sim)."""
    m, d = expert.m, expert.d
    tip_z = None
    palm_z = d.xpos[expert.eef_bid][2]
    for g in range(m.ngeom):
        bid = m.geom_bodyid[g]
        bname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
        if "finger" in bname:
            gz = d.geom_xpos[g][2]
            half = m.geom_size[g][1] if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE else 0.0
            z_bottom = gz - half
            tip_z = z_bottom if tip_z is None else min(tip_z, z_bottom)
    return palm_z, tip_z


def calib():
    env = make_env()
    expert = IKExpert(env)
    env.reset()
    expert.rebind(env)
    env._grip_open_max = env.gripper_opening()
    b = env.bottle_pos()
    t = env.tray_center()
    print("bottle pos:", np.round(b, 4).tolist())
    print("tray   pos:", np.round(t, 4).tolist())
    print("finger open total qpos:", round(env.gripper_opening(), 4))

    # measured in-sim max aperture: drive EE above the bottle, open wide
    q_arm, err = expert.solve_ik(np.array([b[0], b[1], b[2] + 0.12]))
    print("hover IK residual:", round(err, 4))

    # descend to grasp band and read palm/fingertip vs bottle geometry
    z_grasp = b[2] + GRASP_BAND_OFF
    q_arm, err = expert.solve_ik(np.array([b[0], b[1], z_grasp]))
    print("grasp IK residual:", round(err, 4))
    for _ in range(300):
        qa = expert.d.qpos[expert.arm_adrs]
        if np.all(np.abs(q_arm - qa) < ARRIVE_TOL):
            break
        a = np.zeros(env.action_dim)
        a[:7] = qa + np.clip(q_arm - qa, -DQ_MAX, DQ_MAX)
        a[7] = -1.0
        env.step(a)
    palm_z, tip_z = _fingertip_stats(expert)
    print("EE target z=%.4f | physical eef-body z=%.4f | fingertip bottom z=%.4f"
          % (z_grasp, palm_z, tip_z))
    print("  fingertip bottom - bottle center = %+.4f (band: want ~[-0.02, +0.03])"
          % (tip_z - b[2]))
    print("  fingertip bottom - table = %.4f (palm clearance check)"
          % (tip_z - env.table_top_z))
    print("  effective eef->tip drop = %.4f" % (palm_z - tip_z))

    # close on the bottle and read the grasping finger qpos
    hold = np.zeros(env.action_dim)
    hold[:7] = expert.d.qpos[expert.arm_adrs]
    hold[7] = 1.0
    for _ in range(45):
        env.step(hold)
    print("finger qpos while grasping:", round(env.gripper_opening(), 4))
    print("CALIB_OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["calib", "test", "demo"])
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.mode == "calib":
        calib()
        sys.exit(0)

    import imageio
    env = make_env()
    expert = IKExpert(env)
    n = args.episodes
    wins = 0
    saved = 0
    for ep in range(n):
        env.reset()
        frames = []

        def hook(jstep, o, _f=frames):
            if args.mode == "demo" and saved == 0:
                _f.append(o)   # placeholder; real render below

        if args.mode == "demo" and saved == 0:
            ren = mujoco.Renderer(env.sim.model._model, height=480, width=640)

            def hook(jstep, o, _f=frames, _ren=ren, _env=env):
                cam = mujoco.MjvCamera()
                cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                cam.fixedcamid = mujoco.mj_name2id(
                    _env.sim.model._model, mujoco.mjtObj.mjOBJ_CAMERA, "agentview")
                _ren.update_scene(data=_env.sim.data._data, camera=cam)
                _f.append(_ren.render())

        ok, steps, info = expert.run(env, record_hook=hook, verbose=True)
        wins += int(ok)
        bottle_end = env.bottle_pos()
        print(f"[episode {ep}] success={ok} steps={steps} info={info} "
              f"bottle_end={np.round(bottle_end, 3).tolist()} "
              f"tray={np.round(env.tray_center(), 3).tolist()}")
        if ok and args.mode == "demo" and saved == 0 and frames:
            os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
            imageio.mimsave(args.out, frames, fps=int(env.control_freq))
            print("demo video ->", args.out)
            saved += 1
    rate = wins / max(n, 1)
    print(f"SUCCESS_RATE={wins}/{n} = {rate:.0%}")
    sys.exit(0 if rate >= 0.8 else 1)

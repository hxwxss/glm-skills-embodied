# -*- coding: utf-8 -*-
"""
collect_demos.py -- M4: capture the demonstration (HDF5 + MP4)
==============================================================

Runs the IK expert until it produces a successful episode (default: exactly
one -- the requested demonstration), records LIBERO/robosuite-style data and
writes:

  demos/demo.hdf5
    data/demo_0
      attrs: num_samples, success, instruction, model xml string
      actions            (T, 8)    absolute joint targets (7) + gripper (1)
      obs/
        agentview_image   (T, 256, 256, 3)
        robot0_eef_pos    (T, 3)   robot0_eef_quat (T, 4)
        robot0_joint_pos  (T, 7)   robot0_gripper_qpos (T, 2)
        lid_qpos          (T,)     extra observation: hinge angle
      dones              (T,)
    attrs: spec_snapshot (full IR), success_rate, total_episodes

  rollouts/lid_open_demo.mp4   side-by-side agentview+sideview, 640x480 @20fps

The HDF5 is read back and verified (shapes + success + a mid-episode frame
dumped to renders/) before exiting 0.

Usage:
    python collect_demos.py --episodes 1 --out ../demos/demo.hdf5
"""

import argparse
import json
import os
import sys

import h5py
import numpy as np

import mjrender as R
from expert_ik import IKExpert, make_env
from task_open_lid import DEFAULT_SPEC

HERE = os.path.dirname(os.path.abspath(__file__))


def run_one(env, ex, ren, cams, record_video):
    """Execute one episode, recording data (+ frames). Returns dict."""
    obs = env.reset()
    ex.rebind(env)
    frames = {"actions": [], "agentview_image": [], "robot0_eef_pos": [],
              "robot0_eef_quat": [], "robot0_joint_pos": [],
              "robot0_gripper_qpos": [], "lid_qpos": [], "states": [], "dones": []}
    vid = []

    def hook(jstep, o, action):
        frames["actions"].append(np.array(action, dtype=float))
        frames["agentview_image"].append(np.array(o["agentview_image"]))
        frames["robot0_eef_pos"].append(np.array(o["robot0_eef_pos"]))
        frames["robot0_eef_quat"].append(np.array(o["robot0_eef_quat"]))
        frames["robot0_joint_pos"].append(np.array(o["robot0_joint_pos"]))
        frames["robot0_gripper_qpos"].append(np.array(o["robot0_gripper_qpos"]))
        frames["lid_qpos"].append(np.array(env.lid_angle()))
        frames["states"].append(_np.concatenate([_np.asarray(env.sim.get_state().qpos, dtype=float), _np.asarray(env.sim.get_state().qvel, dtype=float)]))
        frames["dones"].append(False)
        if record_video:
            a = R.render_snapshot(ren, env, cams["agentview"])
            s = R.render_snapshot(ren, env, cams["sideview"])
            vid.append(np.concatenate([a, s], axis=1))

    # replicate run() with per-step recording (expert.run has no per-step
    # obs hook for actions, so inline the same logic)
    import numpy as _np
    wps = ex.plan()
    prev_q = None
    done = False
    for wp in wps:
        if done:
            break
        if wp["pos"] is not None:
            q_arm, _, _ = ex.solve_ik(wp["pos"], seed=prev_q)
            prev_q = q_arm
            held = 0
            for _ in range(400):
                qa = ex.d.qpos[ex.arm_adrs]
                a = _np.zeros(env.action_dim)
                a[:7] = qa + _np.clip(q_arm - qa, -0.25, 0.25)
                a[7] = float(wp["gripper"])
                obs, _, done, _ = env.step(a)
                hook(len(frames["dones"]), obs, a)
                if _np.all(np.abs(q_arm - qa) < 0.03):
                    held += 1
                    if held >= 15:
                        break
                else:
                    held = 0
                if done:
                    break
            dwell = int(wp.get("dwell") or 0)
            if dwell and not done:
                hold = _np.zeros(env.action_dim)
                hold[:7] = ex.d.qpos[ex.arm_adrs]
                hold[7] = float(wp["gripper"])
                for _ in range(dwell):
                    obs, _, done, _ = env.step(hold)
                    hook(len(frames["dones"]), obs, a)
                    if done:
                        break
        else:
            hold = _np.zeros(env.action_dim)
            hold[:7] = ex.d.qpos[ex.arm_adrs]
            hold[7] = float(wp["gripper"])
            for _ in range(int(wp.get("dwell") or 30)):
                obs, _, done, _ = env.step(hold)
                hook(len(frames["dones"]), obs, hold)
                if done:
                    break

    success = bool(env._check_success())
    frames["dones"][-1] = True
    return {
        "success": success,
        "steps": len(frames["dones"]),
        "lid_end": env.lid_angle(),
        "data": {k: np.array(v) for k, v in frames.items()},
        "video": vid,
    }


def verify_readback(path, expect_episodes):
    """Read the HDF5 back and verify schema + shapes."""
    ok = True
    with h5py.File(path, "r") as f:
        grp = f["data"]
        n = len(grp)
        assert n == expect_episodes, "episode count %d != %d" % (n,
                                                                 expect_episodes)
        for i in range(n):
            g = grp["demo_%d" % i]
            T = int(g.attrs["num_samples"])
            if not bool(g.attrs["success"]):
                print("  [FAIL] demo_%d success=False" % i)
                ok = False
            checks = [
                ("actions", (T, 8)),
                ("obs/agentview_image", (T, 256, 256, 3)),
                ("obs/robot0_eef_pos", (T, 3)),
                ("obs/robot0_eef_quat", (T, 4)),
                ("obs/robot0_joint_pos", (T, 7)),
                ("obs/robot0_gripper_qpos", (T, 2)),
                ("obs/lid_qpos", (T,)),
                ("dones", (T,)),
            ]
            for key, shape in checks:
                got = g[key].shape
                if got != shape:
                    print("  [FAIL] demo_%d/%s shape %s != %s"
                          % (i, key, got, shape))
                    ok = False
            mid = T // 2
            frame = g["obs/agentview_image"][mid]
            out = os.path.join(HERE, "..", "renders",
                               "hdf5_midframe_demo_%d.png" % i)
            from PIL import Image
            Image.fromarray(frame).save(out)
            print("  demo_%d: T=%d success=%s lid_end=%.2f mid-frame -> %s"
                  % (i, T, bool(g.attrs["success"]),
                     float(g["obs/lid_qpos"][-1]), os.path.basename(out)))
        print("  spec snapshot present:", "spec_snapshot" in f.attrs)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--out", default=os.path.join(HERE, "..", "demos",
                                                  "demo.hdf5"))
    ap.add_argument("--video", default=os.path.join(HERE, "..", "rollouts",
                                                    "lid_open_demo.mp4"))
    ap.add_argument("--max-attempts", type=int, default=8)
    args = ap.parse_args()

    import imageio
    env = make_env(camera_obs=True, height=256)
    ex = IKExpert(env)
    ren = R.make_renderer(env, height=480, width=640)
    cams = {n: R.camera_fixed(env, n) for n in ("agentview", "sideview")}

    episodes = []
    wins = attempts = 0
    videos_saved = 0
    while len(episodes) < args.episodes and attempts < args.max_attempts:
        attempts += 1
        rec = run_one(env, ex, ren, cams, record_video=(videos_saved == 0))
        wins += int(rec["success"])
        print("[attempt %d] success=%s steps=%d lid_end=%.3f"
              % (attempts, rec["success"], rec["steps"], rec["lid_end"]))
        if rec["success"]:
            episodes.append(rec)
            if videos_saved == 0 and rec["video"]:
                os.makedirs(os.path.dirname(os.path.abspath(args.video)),
                            exist_ok=True)
                imageio.mimsave(args.video, rec["video"],
                                fps=int(env.control_freq))
                print("demo video ->", args.video,
                      "(%d frames)" % len(rec["video"]))
                videos_saved += 1
    env.close()

    if len(episodes) < args.episodes:
        print("FAILED: only %d/%d successful episodes in %d attempts"
              % (len(episodes), args.episodes, attempts))
        sys.exit(1)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(DEFAULT_SPEC, encoding="utf-8") as fh:
        spec = json.load(fh)
    with h5py.File(args.out, "w") as f:
        f.attrs["spec_snapshot"] = json.dumps(spec)
        f.attrs["success_rate"] = wins / max(attempts, 1)
        f.attrs["total_attempts"] = attempts
        f.attrs["total_episodes"] = len(episodes)
        f.attrs["instruction"] = spec["task"]["instruction"]
        grp = f.create_group("data")
        for i, ep in enumerate(episodes):
            g = grp.create_group("demo_%d" % i)
            g.attrs["num_samples"] = ep["steps"]
            g.attrs["success"] = ep["success"]
            g.attrs["instruction"] = spec["task"]["instruction"]
            g.create_dataset("actions", data=ep["data"]["actions"],
                             compression="gzip")
            obs_g = g.create_group("obs")
            for k in ("agentview_image", "robot0_eef_pos", "robot0_eef_quat",
                      "robot0_joint_pos", "robot0_gripper_qpos", "lid_qpos"):
                obs_g.create_dataset(k, data=ep["data"][k],
                                     compression="gzip")
            if "states" in ep["data"]:
                g.create_dataset("states", data=ep["data"]["states"],
                                 compression="gzip")
            g.create_dataset("dones", data=ep["data"]["dones"].astype(np.uint8))
            meta = g.create_group("env_args")
            meta.attrs["env_name"] = "OpenBoxLid"
            meta.attrs["spec_json"] = json.dumps(spec)
            if hasattr(env, 'model') and hasattr(env.model, 'get_xml'):
                g.attrs["model_file"] = env.model.get_xml()

    print("WROTE %s (%d episodes, raw success %d/%d attempts)"
          % (args.out, len(episodes), wins, attempts))

    if not verify_readback(args.out, len(episodes)):
        print("READBACK VERIFICATION FAILED")
        sys.exit(1)
    print("READBACK OK")
    sys.exit(0)


if __name__ == "__main__":
    main()

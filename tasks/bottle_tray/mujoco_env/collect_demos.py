# -*- coding: utf-8 -*-
"""
collect_demos.py — M4: LIBERO/robosuite-style HDF5 demonstration dataset
=========================================================================

Runs the IK expert for N episodes. Per step records:

    data/demo_i/
      attrs: num_samples, success, instruction
      actions            (T,8)   absolute joint targets (7) + gripper (1)
      obs/
        agentview_image   (T,256,256,3)   mujoco.Renderer on the LIVE model
        robot0_eef_pos    (T,3)   robot0_eef_quat (T,4)
        robot0_joint_pos  (T,7)   robot0_gripper_qpos (T,2)
      dones              (T,)
      env_args attrs (env name + full IR snapshot)

Only success=True episodes are written. Also renders one 640x480 MP4 of the
first successful episode for human review (RGB straight into imageio — no
double color conversion).

Usage:
    python collect_demos.py --episodes 8 --out ../demos/bottle_tray_demo.hdf5
"""

import argparse
import json
import os

import h5py
import numpy as np
import mujoco

from expert_ik import IKExpert, make_env

HERE = os.path.dirname(os.path.abspath(__file__))

IMG_SIZE = 256          # dataset camera resolution
VIDEO_SIZE = (480, 640)  # MP4 review resolution as (height, width)


def collect(episodes, out_path, video_path):
    env = make_env(camera_obs=False, height=IMG_SIZE)
    expert = IKExpert(env)
    episodes_data = []
    wins = 0
    video_saved = False

    for ep in range(episodes):
        env.reset()
        expert.rebind(env)
        # bind renderers to the LIVE model (hard_reset rebuilds it)
        ren_obs = mujoco.Renderer(env.sim.model._model,
                                  height=IMG_SIZE, width=IMG_SIZE)
        ren_vid = None if video_saved else mujoco.Renderer(
            env.sim.model._model, height=VIDEO_SIZE[0], width=VIDEO_SIZE[1])

        def snap(ren):
            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            cam.fixedcamid = mujoco.mj_name2id(
                env.sim.model._model, mujoco.mjtObj.mjOBJ_CAMERA, "agentview")
            ren.update_scene(data=env.sim.data._data, camera=cam)
            return ren.render()   # RGB

        bottle0 = env.bottle_pos().copy()
        tray0 = env.tray_center().copy()
        frames = {"actions": [], "agentview_image": [], "robot0_eef_pos": [],
                  "robot0_eef_quat": [], "robot0_joint_pos": [],
                  "robot0_gripper_qpos": [], "dones": []}
        video_frames = []

        wps = expert.plan(bottle0, tray0)
        jsteps = 0
        done = False

        def step_rec(a):
            nonlocal jsteps, done
            obs, _, done, _ = env.step(a)
            jsteps += 1
            frames["actions"].append(np.array(a, dtype=float))
            frames["agentview_image"].append(snap(ren_obs))
            if ren_vid is not None:
                video_frames.append(snap(ren_vid))
            frames["robot0_eef_pos"].append(np.array(obs["robot0_eef_pos"]))
            frames["robot0_eef_quat"].append(np.array(obs["robot0_eef_quat"]))
            frames["robot0_joint_pos"].append(np.array(obs["robot0_joint_pos"]))
            frames["robot0_gripper_qpos"].append(
                np.array(obs["robot0_gripper_qpos"]))
            frames["dones"].append(bool(done))

        for wp in wps:
            if wp["pos"] is not None:
                q_arm, err = expert.solve_ik(wp["pos"])
                dq_cap = float(wp.get("dq", 0.25))
                for _ in range(400):
                    qa = expert.d.qpos[expert.arm_adrs]
                    if np.all(np.abs(q_arm - qa) < 0.06):
                        break
                    a = np.zeros(env.action_dim)
                    a[:7] = qa + np.clip(q_arm - qa, -dq_cap, dq_cap)
                    a[7] = float(wp["gripper"])
                    step_rec(a)
                    if done:
                        break
            else:
                hold = np.zeros(env.action_dim)
                hold[:7] = expert.d.qpos[expert.arm_adrs]
                hold[7] = float(wp["gripper"])
                for _ in range(int(wp.get("dwell") or 30)):
                    step_rec(hold)
                    if done:
                        break
            success = env._check_success()
            if done or success:
                break

        success = bool(env._check_success())
        wins += int(success)
        episodes_data.append({
            "actions": np.array(frames["actions"], dtype=float),
            "obs": {k: np.array(v) for k, v in frames.items()
                    if k not in ("actions", "dones")},
            "dones": np.array(frames["dones"], dtype=np.uint8),
            "success": success,
            "steps": jsteps,
            "bottle_start": bottle0.tolist(),
            "bottle_end": env.bottle_pos().tolist(),
            "tray_pos": env.tray_center().tolist(),
        })
        print(f"[episode {ep}] success={success} steps={jsteps} "
              f"transitions={len(frames['dones'])}")

        if success and not video_saved and video_frames:
            os.makedirs(os.path.dirname(os.path.abspath(video_path)),
                        exist_ok=True)
            import imageio
            imageio.mimsave(video_path, video_frames,
                            fps=int(env.control_freq))
            print("review video ->", video_path)
            video_saved = True
        ren_obs.close()
        if ren_vid is not None:
            ren_vid.close()

    good = [e for e in episodes_data if e["success"]]
    failures = [ep for ep in episodes_data if not ep["success"]]
    if failures:
        print(f"[collect] {len(failures)} failed episodes NOT written to the dataset")
    episodes_data = [ep for ep in episodes_data if ep["success"]]
    if not episodes_data:
        print("[collect] no successful episodes — aborting without writing data")
        sys.exit(1)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with h5py.File(out_path, "w") as f:
        f.attrs["spec_snapshot"] = json.dumps(env.spec)
        f.attrs["success_rate"] = len(good) / max(len(episodes_data), 1)
        f.attrs["total_episodes"] = len(episodes_data)
        f.attrs["instruction"] = env.spec["task"]["instruction"]
        grp = f.create_group("data")
        for i, ep in enumerate(good):
            g = grp.create_group(f"demo_{i}")
            g.attrs["num_samples"] = len(ep["dones"])
            g.attrs["success"] = ep["success"]
            g.attrs["instruction"] = env.spec["task"]["instruction"]
            g.create_dataset("actions", data=ep["actions"], compression="gzip")
            obs_g = g.create_group("obs")
            for k, v in ep["obs"].items():
                if len(v):
                    obs_g.create_dataset(k, data=v, compression="gzip")
            g.create_dataset("dones", data=ep["dones"])
            meta = g.create_group("env_args")
            meta.attrs["env_name"] = "PutBottleInTray"
            meta.attrs["spec_json"] = json.dumps(env.spec)
    print(f"WROTE {out_path}  success_rate={len(good)}/{len(episodes_data)}")
    env.close()
    return len(good), len(episodes_data)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--out", default=os.path.join(
        HERE, "..", "demos", "bottle_tray_demo.hdf5"))
    ap.add_argument("--video", default=os.path.join(
        HERE, "..", "rollouts", "bottle_in_tray_demo.mp4"))
    args = ap.parse_args()
    good, total = collect(args.episodes, args.out, args.video)
    # M4 gate: every written episode must be a success
    sys_exit = 0 if good == total and good >= 6 else 1
    import sys
    sys.exit(sys_exit)

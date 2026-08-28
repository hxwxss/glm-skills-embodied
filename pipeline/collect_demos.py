# -*- coding: utf-8 -*-
"""
collect_demos.py — M4:采集 LIBERO/robosuite 风格 HDF5 演示数据集
================================================================

跑 IK 专家 N 个 episode,每一步记录:

    data/demo_i/
      attrs: num_samples, model_file(内嵌 xml), success, instruction
      actions        (T,8)   绝对关节目标(7) + 夹爪(1)
      obs/
        agentview_image (T,256,256,3)
        robot0_eef_pos  (T,3)   robot0_eef_quat (T,4)
        robot0_joint_pos (T,7)  robot0_gripper_qpos (T,2)
      dones          (T,)

并写出 dataset 级元数据(spec 快照、成功率)。

用法:
    python collect_demos.py --episodes 10 --out ../data/demo.hdf5
"""

import argparse
import json
import os

import h5py
import sys

import numpy as np

from expert_ik import IKExpert, make_env

HERE = os.path.dirname(os.path.abspath(__file__))


def collect(episodes, out_path, camera=True):
    import mujoco as _mj
    from expert_ik import absolute_controller_config
    # env_args 存"构造时传入"的 composite 控制器配置;
    # reset 后 env 内部的规范化 dict 含内部键,原样传回会重建失败
    controller_cfg = absolute_controller_config()
    env = make_env(camera_obs=camera, controller_configs=controller_cfg)
    expert = IKExpert(env)
    ren = _mj.Renderer(env.sim.model._model, height=480, width=640)
    cam = _mj.MjvCamera()
    cam.type = _mj.mjtCamera.mjCAMERA_FIXED
    cam.fixedcamid = _mj.mj_name2id(env.sim.model._model,
                                    _mj.mjtObj.mjOBJ_CAMERA, "agentview")

    def snap():
        ren.update_scene(data=env.sim.data._data, camera=cam)
        return ren.render()  # RGB

    episodes_data = []
    wins = 0
    for ep in range(episodes):
        env.reset()
        expert.rebind(env)
        cam.fixedcamid = _mj.mj_name2id(env.sim.model._model,
                                        _mj.mjtObj.mjOBJ_CAMERA, "agentview")
        cube0 = np.array(env.sim.data.xpos[env.obj_body_id["RedCube"]]).copy()

        frames = {"actions": [], "agentview_image": [], "robot0_eef_pos": [],
                  "robot0_eef_quat": [], "robot0_joint_pos": [],
                  "robot0_gripper_qpos": [], "states": [], "dones": []}
        success = False

        # LIBERO drop-in: full MJCF + env_args captured right after reset
        model_xml = env.model.get_xml()
        env_args = {
            "env_name": "PutRedInBox",
            "env_type": 0,
            "env_kwargs": {
                "robots": ["Panda"],
                # 完整 composite 控制器配置:重建时保持绝对关节位置语义
                "controller_configs": controller_cfg,
                "control_freq": int(env.control_freq),
                "has_renderer": False,
                "has_offscreen_renderer": True,
                "use_camera_obs": False,
            },
            "init_seed": None,
            # 机器人基座世界位姿(重建时 set_base_xpos 恢复)
            "base_pos": [float(v) for v in env.panda_base_pos],
        }

        def sim_state_vec():
            st = env.sim.get_state()
            parts = [np.asarray(st.qpos, dtype=float).ravel(),
                     np.asarray(st.qvel, dtype=float).ravel()]
            act = getattr(st, "act", None)
            if act is not None:
                parts.append(np.asarray(act, dtype=float).ravel())
            return np.concatenate(parts)

        # 手动逐步执行以同步记录(与 expert.run 相同逻辑)
        wps = expert.plan(cube0)
        jsteps = 0
        done = False

        def step_rec(a):
            nonlocal jsteps, done
            obs, _, done, _ = env.step(a)
            jsteps += 1
            frames["actions"].append(np.array(a, dtype=float))
            frames["agentview_image"].append(snap())
            frames["robot0_eef_pos"].append(np.array(obs["robot0_eef_pos"]))
            frames["robot0_eef_quat"].append(np.array(obs["robot0_eef_quat"]))
            frames["robot0_joint_pos"].append(np.array(obs["robot0_joint_pos"]))
            frames["robot0_gripper_qpos"].append(np.array(obs["robot0_gripper_qpos"]))
            frames["states"].append(sim_state_vec())
            frames["dones"].append(bool(done))
        for wp_i, wp in enumerate(wps):
            if wp["pos"] is not None:
                q_arm, err = expert.solve_ik(wp["pos"])
                for _ in range(300):
                    qa = expert.d.qpos[expert.arm_adrs]
                    if np.all(np.abs(q_arm - qa) < 0.06):
                        break
                    a = np.zeros(env.action_dim)
                    a[:7] = qa + np.clip(q_arm - qa, -0.25, 0.25)
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

        # LIBERO convention: the final transition of every episode is a done
        if len(frames["dones"]):
            frames["dones"][-1] = True

        wins += int(success)
        cube_end = np.array(env.sim.data.xpos[env.obj_body_id["RedCube"]])
        episodes_data.append({
            "actions": np.array([a for a in frames["actions"] if a is not None]),
            "obs": {k: np.array(v) for k, v in frames.items()
                    if k not in ("actions", "dones")},
            "dones": np.array(frames["dones"], dtype=np.uint8),
            "states": np.array(frames["states"], dtype=float),
            "model_file": model_xml,
            "env_args": env_args,
            "success": bool(success),
            "steps": jsteps,
            "cube_start": cube0.tolist(),
            "cube_end": cube_end.tolist(),
            "instruction": env.spec["task"]["instruction"],
            "model_xml": env.sim.model._model.to_xml()
            if hasattr(env.sim.model._model, "to_xml") else None,
        })
        print(f"[episode {ep}] success={success} steps={jsteps} "
              f"transitions={len(frames['dones'])}")

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
        f.attrs["success_rate"] = wins / max(len(episodes_data), 1)
        f.attrs["total_episodes"] = len(episodes_data)
        grp = f.create_group("data")
        for i, ep in enumerate(episodes_data):
            g = grp.create_group(f"demo_{i}")
            g.attrs["num_samples"] = len(ep["dones"])
            g.attrs["success"] = ep["success"]
            g.attrs["instruction"] = ep["instruction"]
            g.attrs["model_file"] = ep["model_file"]
            g.attrs["env_args"] = json.dumps(ep["env_args"])
            g.create_dataset("states", data=ep["states"], compression="gzip")
            g.create_dataset("actions", data=ep["actions"], compression="gzip")
            obs_g = g.create_group("obs")
            for k, v in ep["obs"].items():
                if len(v):
                    obs_g.create_dataset(k, data=v, compression="gzip")
            g.create_dataset("dones", data=ep["dones"])
            # env 复原所需的环境参数
            meta = g.create_group("env_args")
            meta.attrs["env_name"] = "PutRedInBox"
            meta.attrs["spec_json"] = json.dumps(env.spec)
    print(f"WROTE {out_path}  success_rate={wins}/{episodes}")
    if wins < episodes:
        sys.exit(1)
    env.close()


def _record(frames, obs, action, done):
    frames["actions"].append(np.array(action, dtype=float))
    frames["dones"].append(bool(done))
    for k in ("agentview_image",):
        if k in obs:
            frames[k].append(np.array(obs[k]))
    for k in ("robot0_eef_pos", "robot0_eef_quat", "robot0_joint_pos",
              "robot0_gripper_qpos"):
        if k in obs:
            frames[k].append(np.array(obs[k]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--out", default=os.path.join(HERE, "..", "demos", "demo.hdf5"))
    args = ap.parse_args()
    collect(args.episodes, args.out)

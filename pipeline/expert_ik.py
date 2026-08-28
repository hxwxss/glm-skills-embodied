# -*- coding: utf-8 -*-
"""
expert_ik.py — 离线/在线混合 IK 专家（M3 最终方案）
====================================================

用户提议的经典路线，也是本管线最终采用的方案：

  物体级关键点（世界系）  →  mink 数值 IK 逐点求关节构型  →  关节空间跟踪执行
                              （每段起点绑实时状态 = 闭环重定基准）

执行层使用 robosuite 的 JOINT_POSITION 控制器（绝对关节位置目标），
完全不依赖 OSC 的语义差异；夹爪动作独立于末端轨迹。

用法：
    python expert_ik.py calib                 # 单程走到 grasp 点并渲染
    python expert_ik.py test --episodes 10    # 成功率统计（验收 ≥80%）
    python expert_ik.py demo --out demo.mp4   # 录制演示视频
"""

import argparse
import os
import sys

import numpy as np
import mujoco
import mink

from task_put_red_in_box import PutRedInBox

HERE = os.path.dirname(os.path.abspath(__file__))

APPROACH_H = 0.12         # 抓取悬停高度(相对方块中心)
GRASP_DROP_EXTRA = 0.005  # 下探补偿(指尖接触面估计误差)
LIFT_H = 0.16             # 提升高度
OVER_LIP_MARGIN = 0.04    # 越过盒口余量
DQ_MAX = 0.25             # 关节空间单步最大增量(rad)
SETTLE = 8                # 到位后稳定帧数


def joint_controller_config():
    """BASIC composite 配置,右臂段替换为绝对关节位置控制器."""
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
    # 关节弧度可达 ±π 以上,不放开则动作被 env 的 ±1 钳位静默截断
    right["input_max"] = [4.5] * 7
    right["input_min"] = [-4.5] * 7
    return cfg


def make_env(camera_obs=False, height=512, horizon=1400):
    # RethinkMount 版本经过四点可达性审计(err<5mm),保持一致以保证可复现
    return PutRedInBox(
        has_renderer=False,
        has_offscreen_renderer=camera_obs,
        use_camera_obs=camera_obs,
        controller_configs=joint_controller_config(),
        base_types=["RethinkMount"],
        camera_names=["agentview"],
        camera_heights=height,
        camera_widths=height,
        horizon=horizon,
    )


class IKExpert:
    def __init__(self, env):
        self.env = env
        sp = env.spec
        zone = next(o for o in sp["objects"]
                    if "container" in o.get("semantic", []))["inner_zone"]
        self.zone_center = np.array(zone["pos"])
        self.zone_half = np.array(zone["size"]) / 2
        red = next(o for o in sp["objects"] if o["id"] == "Prop_Cube_Red")
        self.cube_half = red["dims"][0] / 2
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
            position_cost=10.0, orientation_cost=0.3, lm_damping=1.0)
        R_t = np.eye(3); R_t[1, 1] = -1; R_t[2, 2] = -1   # 手指朝下
        self.grasp_rot = mink.SO3.from_matrix(R_t)
        # 记录 reset 位形(用于 IK 局部极小重试)
        arm_save = self.d.qpos[self.arm_adrs].copy()
        mujoco.mj_forward(self.m, self.d)
        self.q_home_arm = arm_save

    def sync(self):
        """把物理仿真当前 qpos 写入 mink 配置（闭环重定基准）。"""
        self.cfg.data.qpos[:] = self.d.qpos
        self.cfg.data.time = self.d.time
        mujoco.mj_forward(self.m, self.cfg.data)

    def eef_pos(self):
        return np.array(self.env.sim.data.xpos[self.eef_bid])

    def rebind(self, env):
        """每次 reset 后重绑 model/data(mink Configuration 绑定具体模型实例)."""
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
        R_t = np.eye(3); R_t[1, 1] = -1; R_t[2, 2] = -1
        self.grasp_rot = mink.SO3.from_matrix(R_t)

    def solve_ik(self, pos, max_iters=400, tol=6e-3):
        """以当前构型为起点迭代求解到目标位姿,失败则从 reset home 重试.
        返回 (q_arm, 残差)."""
        T = mink.SE3.from_rotation_and_translation(
            self.grasp_rot, np.asarray(pos, dtype=float))
        self.task.set_target(T)

        def iterate():
            err = float("inf")
            for _ in range(max_iters):
                dv = mink.solve_ik(self.cfg, [self.task], dt=1 / 60,
                                   solver="quadprog")
                self.cfg.integrate_inplace(dv, 1 / 60)
                cur = self.cfg.data.xpos[self.eef_bid]
                err = float(np.linalg.norm(cur - pos))
                if err < tol:
                    break
            return err

        self.sync()
        err = iterate()
        if err > tol:
            # 局部极小:从 reset 关节位形重新起步再解一次
            self.cfg.data.qpos[:] = self.d.qpos   # 物理态
            self.cfg.data.qpos[self.arm_adrs] = self.q_home_arm
            self.cfg.data.qvel[:] = 0
            mujoco.mj_forward(self.m, self.cfg.data)
            err = iterate()
        q_arm = self.cfg.data.qpos[self.arm_adrs].copy()
        return q_arm, err

    def plan(self, cube_pos):
        cx, cy, cz = cube_pos
        tx, ty, _ = self.zone_center
        z_lip = self.table_top_z + 2 * self.cube_half + OVER_LIP_MARGIN
        z_over = z_lip + 0.02
        W = [
            dict(pos=np.array([cx, cy, cz + APPROACH_H]), gripper=-1,
                 note="hover-above-cube"),
            # 实测掌心存在 ~2.3cm 重力下垂:IK 目标=方块中心时,
            # 物理掌心恰在中心附近,指尖咬住方块中部(台顶不碰)
            dict(pos=np.array([cx, cy, cz]), gripper=-1,
                 note="descend-to-grasp"),
            dict(pos=None, gripper=1, note="close-gripper", dwell=35),
            dict(pos=np.array([cx, cy, cz + LIFT_H]), gripper=1, note="lift"),
            dict(pos=np.array([tx, ty, max(cz + LIFT_H, z_over)]),
                 gripper=1, note="move-over-box"),
            dict(pos=np.array([tx, ty, z_over]), gripper=1,
                 note="align-over-mouth", dwell=5),
            dict(pos=np.array([tx, ty, z_lip - 0.01]), gripper=1,
                 note="lower-inside"),
            dict(pos=None, gripper=-1, note="release", dwell=25),
            dict(pos=np.array([tx, ty, z_over + 0.05]), gripper=-1,
                 note="retreat"),
        ]
        return W

    def run(self, env, max_steps=1600, record_hook=None, verbose=False):
        """逐相位执行(与已验证的诊断循环逐字一致)."""
        # robosuite hard_reset 会重建 MjModel,必须重新绑定 mink 配置
        self.rebind(env)
        cube_id = env.obj_body_id["RedCube"]
        cube0 = np.array(env.sim.data.xpos[cube_id]).copy()
        wps = self.plan(cube0)
        jsteps = 0

        for wp_i, wp in enumerate(wps):
            note = wp["note"]
            if wp["pos"] is not None:
                q_arm, err = ex_solve(self, wp["pos"])
                for _ in range(300):
                    qa = self.d.qpos[self.arm_adrs]
                    if np.all(np.abs(q_arm - qa) < 0.06):
                        break
                    a = np.zeros(env.action_dim)
                    a[:7] = qa + np.clip(q_arm - qa, -0.25, 0.25)
                    a[7] = float(wp["gripper"])
                    obs, _, done, _ = env.step(a)
                    jsteps += 1
                    if record_hook is not None:
                        record_hook(jsteps, obs)
                    if done:
                        return False, jsteps, {"phase": note}
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
                        return False, jsteps, {"phase": note}
            if env._check_success():
                return True, jsteps, {"phase": note}
        return env._check_success(), jsteps, {"phase": wps[-1]["note"]}


def ex_solve(expert, pos):
    """诊断版同款 solve_ik 调用(不随 episode 状态缓存)."""
    return expert.solve_ik(pos)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["calib", "test", "demo"])
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import imageio
    demos_saved = 0
    wins = 0

    if args.mode == "calib":
        env = make_env()
        ex = IKExpert(env)
        env.reset()
        cube = np.array(env.sim.data.xpos[env.obj_body_id["RedCube"]])
        wps = ex.plan(cube)
        tgt = wps[1]["pos"]
        q, err = ex.solve_ik(tgt)
        print("grasp IK residual:", round(err, 4), "q:", np.round(q, 2).tolist())
        sys.exit(0)

    import cv2 as _cv2
    import mujoco as _mj
    cv2 = _cv2

    env = make_env(camera_obs=False)
    expert = IKExpert(env)
    n = args.episodes

    # robosuite 的 obs 渲染管线存在 stale-model 问题(红盒缺失),
    # 演示视频统一走 mujoco.Renderer 直渲 agentview 相机
    ren = _mj.Renderer(env.sim.model._model, height=480, width=640)
    cam = _mj.MjvCamera()
    cam.type = _mj.mjtCamera.mjCAMERA_FIXED
    cam.fixedcamid = _mj.mj_name2id(env.sim.model._model,
                                    _mj.mjtObj.mjOBJ_CAMERA, "agentview")
    def snap():
        ren.update_scene(data=env.sim.data._data, camera=cam)
        return ren.render()   # imageio 期望 RGB,保持原样

    frames = []
    def record_enabled():
        return args.mode == "demo" and (demos_saved == 0)

    for ep in range(n):
        env.reset()
        expert.rebind(env)
        cam.fixedcamid = _mj.mj_name2id(env.sim.model._model,
                                        _mj.mjtObj.mjOBJ_CAMERA, "agentview")
        frames = []

        def hook(jstep, o, _frames=frames):
            if args.mode == "demo" and demos_saved == 0:
                _frames.append(snap())

        ok, steps, info = expert.run(env, record_hook=hook)
        wins += int(ok)
        print(f"[episode {ep}] success={ok} steps={steps} info={info}")
        if ok and args.out and demos_saved == 0 and frames:
            os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
            imageio.mimsave(args.out, frames, fps=int(env.control_freq))
            print("demo video ->", args.out)
            demos_saved += 1
    rate = wins / max(n, 1)
    print(f"SUCCESS_RATE={wins}/{n} = {rate:.0%}")
    sys.exit(0 if rate >= 0.8 else 1)

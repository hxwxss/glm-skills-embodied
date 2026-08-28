# -*- coding: utf-8 -*-
"""
expert_keypoints.py — keypoint 专家策略（M3）
=============================================

思路（MimicGen 同款）：每个 episode 根据红块的当前随机初始位姿生成一组
路径关键点，在关键点之间做匀速直线插值，按 delta-OSC 每步发出 ±5cm 内的
位移增量；夹爪在“贴近—抬起”处闭合、“落入盒—撤离”处张开。

姿态约定：整条轨迹保持 reset 后的末端朝向不变（rotation delta = 0），
robosuite Panda 默认关节归零位即为俯身向前的工作姿态，可直接抓取。

用法：
    python expert_keypoints.py calib                     # 标定 EE/爪语义
    python expert_keypoints.py test --episodes 20        # 成功率验收
    python expert_keypoints.py demo --out demo.mp4       # 录一条演示视频
"""

import argparse
import math
import os
import sys

import numpy as np

from task_put_red_in_box import PutRedInBox

HERE = os.path.dirname(os.path.abspath(__file__))

# 可调超参（首次调试后固定）
APPROACH_H = 0.12         # 抓取前悬停高度（相对方块中心）
GRASP_Z_OFF = 0.09        # EE site 相对指尖接触面的补偿
LIFT_H = 0.18             # 提升高度
OVER_LIP_MARGIN = 0.03    # 越过盒口的余量
SPEED = 0.08              # EE 平移速率（m/s）
ARRIVE_TOL = 0.012

# 顶部朝下抓取的目标姿态（w,x,y,z）：绕世界 X 轴翻 180°，使末端 -Z 对准桌面
GRASP_QUAT_WXYZ = np.array([0.0, 1.0, 0.0, 0.0])

from scipy.spatial.transform import Rotation as _R

def grasp_rotvec():
    """抓取姿态的 rotvec（scipy xyzw 输入 → 轴角），供 absolute OSC 使用."""
    q = GRASP_QUAT_WXYZ
    return _R.from_quat([q[1], q[2], q[3], q[0]]).as_rotvec()

# absolute OSC 控制器配置：动作 = [世界系绝对位置, 目标姿态 rotvec, 夹爪]
def absolute_controller_config():
    """robosuite composite BASIC 配置，arm 段改为世界系绝对位姿输入."""
    from robosuite import load_composite_controller_config
    cfg = load_composite_controller_config(controller="BASIC", robot="Panda")
    arm = cfg["body_parts"]["right"]
    arm["input_type"] = "absolute"
    arm["input_ref_frame"] = "world"
    # absolute 模式下动作是世界系米制坐标，必须放开 delta 模式的 ±1 归一化钳位
    # （否则 z>1.0 的桌面任务点会被 clip 成 z=1.0，手永远下不去）
    arm["input_max"] = [3.0] * 6
    arm["input_min"] = [-3.0] * 6
    return cfg

ABS_CONTROLLER_CONFIG = None  # 延迟构建（make_env 内调用上面的工厂函数）


class KeypointExpert:
    def __init__(self, env):
        self.env = env
        sp = env.spec
        self.table_top_z = float(sp["workspace"]["table_top_z"])
        zone = next(o for o in sp["objects"]
                    if "container" in o.get("semantic", []))["inner_zone"]
        self.zone_center = np.array(zone["pos"])
        self.zone_half = np.array(zone["size"]) / 2
        self.cube_half = next(o for o in sp["objects"]
                              if o["id"] == "Prop_Cube_Red")["dims"][0] / 2
        self._eef_bid = env.sim.model.body_name2id("gripper0_right_eef")

    def eef_pos(self):
        return np.array(self.env.sim.data.xpos[self._eef_bid])

    def eef_R(self):
        return self.env.sim.data.xmat[self._eef_bid].reshape(3, 3)

    def quat_from_R(self, R):
        """旋转矩阵 → (w,x,y,z)。"""
        m = R
        tr = m[0, 0] + m[1, 1] + m[2, 2]
        if tr > 0:
            s = math.sqrt(tr + 1.0) * 2
            q = [0.25 * s, (m[2, 1] - m[1, 2]) / s,
                 (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
        elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
            q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s,
                 (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
        elif m[1, 1] > m[2, 2]:
            s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
            q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s,
                 0.25 * s, (m[1, 2] + m[2, 1]) / s]
        else:
            s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
            q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s,
                 (m[1, 2] + m[2, 1]) / s, 0.25 * s]
        q = np.array(q)
        return q / np.linalg.norm(q)

    def quat_err_axisangle(self, q_cur, q_des):
        """返回把当前姿态转到目标姿态所需的轴角向量（世界系）。"""
        # 相对旋转 r = q_des * inv(q_cur)
        w0, v0 = q_cur[0], q_cur[1:]
        w1, v1 = q_des[0], q_des[1:]
        w = w1 * w0 + np.dot(v1, v0)
        v = w1 * (-v0) + w0 * v1 + np.cross(v1, v0)
        if w < 0:
            w, v = -w, -v
        ang = 2.0 * math.acos(max(-1.0, min(1.0, w)))
        axis = v / (np.linalg.norm(v) + 1e-12)
        return axis * ang

    # ------------------------------------------------------------------
    def plan(self, cube_pos):
        cx, cy, cz = cube_pos
        tx, ty, _ = self.zone_center
        z_lip = (self.table_top_z + 2 * self.cube_half      # cube 顶面参考高度
                 + OVER_LIP_MARGIN)
        z_over_box = z_lip + 0.02
        grasp_z = cz + GRASP_Z_OFF

        W = [
            dict(pos=np.array([cx, cy, cz + APPROACH_H]), gripper=-1,
                 note="hover-above-cube"),
            dict(pos=np.array([cx, cy, grasp_z]), gripper=-1,
                 note="descend-to-grasp"),
            dict(pos=np.array([cx, cy, grasp_z]), gripper=1,
                 note="close-gripper", dwell=30),
            dict(pos=np.array([cx, cy, cz + LIFT_H]), gripper=1,
                 note="lift"),
            dict(pos=np.array([tx, ty, max(cz + LIFT_H, z_over_box)]),
                 gripper=1, note="move-over-box"),
            dict(pos=np.array([tx, ty, z_over_box]), gripper=1,
                 note="align-over-mouth", dwell=8),
            dict(pos=np.array([tx, ty, z_over_box - 0.055]), gripper=1,
                 note="lower-into-box"),
            dict(pos=np.array([tx, ty, z_over_box - 0.075]), gripper=-1,
                 note="release", dwell=18),
            dict(pos=np.array([tx, ty, z_over_box]), gripper=-1,
                 note="retreat"),
        ]
        return W

    # ------------------------------------------------------------------
    def run(self, env, max_steps=600, record_hook=None, verbose=False):
        """执行一个 episode，返回 (success, steps, info)."""
        cube_id = env.obj_body_id["RedCube"]
        waypoints = self.plan(np.array(env.sim.data.xpos[cube_id]))
        dt = 1.0 / env.control_freq
        phase_i, dwell_left = 0, 0
        steps = 0
        rv = grasp_rotvec()

        for step in range(max_steps):
            wp = waypoints[phase_i]
            cur = self.eef_pos()
            delta = wp["pos"] - cur
            dist = float(np.linalg.norm(delta))
            move = min(SPEED * dt, dist)
            action = np.zeros(env.action_dim)
            # absolute OSC：发目标位置（内部会朝目标运动），姿态恒为抓取姿态
            action[:3] = wp["pos"] if dist < 0.25 else cur + delta / dist * move
            action[3:6] = rv
            action[6] = float(wp["gripper"])

            obs, _, done, info = env.step(action)
            steps += 1
            if record_hook is not None:
                record_hook(step, obs)
            if done:
                # robosuite 在 _check_success() 为 True 或到达 horizon 时终止
                return bool(env._check_success()), steps, {"phase": wp["note"]}

            success = env._check_success()
            if verbose and step % 25 == 0:
                print(f"  t={step} phase={wp['note']} dist={dist:.3f} "
                      f"success={success}")

            if dwell_left > 0:
                dwell_left -= 1
                continue
            arrived = dist < ARRIVE_TOL
            if arrived and wp.get("dwell"):
                dwell_left = wp["dwell"]
            if arrived and dwell_left == 0:
                phase_i += 1
                if phase_i >= len(waypoints):
                    break

            if success:
                return True, steps, {"phase": wp["note"]}

        return False, steps, {"phase": waypoints[-1]["note"]}


def make_env(camera_obs=False, height=256):
    return PutRedInBox(
        has_renderer=False,
        has_offscreen_renderer=camera_obs,
        use_camera_obs=camera_obs,
        controller_configs=absolute_controller_config(),
        camera_names=["agentview"],
        camera_heights=height,
        camera_widths=height,
        horizon=700,
    )


def calib():
    """打印 reset 后的关键状态并标定 gripper 动作语义."""
    env = make_env(camera_obs=False)
    env.reset()
    for _ in range(60):
        env.step(np.concatenate([np.zeros(6), [-1.0]]))
    eef = None
    eef = np.array(env.sim.data.xpos[
        env.sim.model.body_name2id("gripper0_right_eef")])
    cube = np.array(env.sim.data.body_xpos[env.obj_body_id["RedCube"]])
    print("eef after hold :", np.round(eef, 3))
    print("cube pos       :", np.round(cube, 3))
    print("eef - cube     :", np.round(eef - cube, 3))

    gq = []
    for j in range(env.sim.model.njnt):
        name = mujoco_name(env.sim, j)
        if "finger" in name:
            adr = env.sim.model.jnt_qposadr[j]
            gq.append((name, round(float(env.sim.data.qpos[adr]), 3)))
    print("finger qpos    :", gq)


def mujoco_name(sim, joint_i):
    # mujoco.MjModel 无 per-joint name getter 时退化为 mj_id2name
    import mujoco
    return mujoco.mj_id2name(sim.model._model,
                             mujoco.mjtObj.mjOBJ_JOINT, joint_i)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["calib", "test", "demo"])
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.mode == "calib":
        calib()
        sys.exit(0)

    import imageio
    env = make_env(camera_obs=args.mode == "demo")
    expert = KeypointExpert(env)
    wins = 0
    saved_demo = False
    for ep in range(args.episodes):
        env.reset()
        frames = []

        def hook(step, o, _f=frames):
            if args.mode == "demo":
                _f.append(o["agentview_image"].copy())

        ok, steps, info = expert.run(env, record_hook=hook)
        wins += int(ok)
        print(f"[episode {ep}] success={ok} steps={steps} ({info})")
        if ok and args.out and not saved_demo:
            os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
            imageio.mimsave(args.out, frames, fps=int(env.control_freq))
            print("saved demo video ->", args.out)
            saved_demo = True
    rate = wins / max(args.episodes, 1)
    print(f"SUCCESS_RATE={wins}/{args.episodes} = {rate:.0%}")
    sys.exit(0 if rate >= 0.8 else 1)

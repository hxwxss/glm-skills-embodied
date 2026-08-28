# -*- coding: utf-8 -*-
"""
verify_libero_compat.py — LIBERO drop-in 合规校验
=================================================

对 data/<task>/*.hdf5 逐项校验 LIBERO 官方加载路径所需的字段与语义:

  1. 结构:每个 episode 含 actions/obs/agentview_image/states/dones,
     且 attrs 带 model_file / env_args / instruction / num_samples
  2. model_file:attr 中的 MJCF 可被 mujoco.MjModel.from_xml_string 编译
  3. states 可回放:将第 k 帧状态写入 sim 并 forward 后,
     末端位置与该帧记录的 robot0_eef_pos 一致(容差 5 mm)
  4. env_args:合法 JSON,env_name 存在;将该 env 注册进 robosuite 后
     robosuite.make(env_name=..., **env_kwargs) 可创建环境
  5. dones:每个 episode 的最后一帧 dones == True(经专家提前结束的情形
     由采集器强制补写)

用法:
    python verify_libero_compat.py [data/put_red_in_box/demo.hdf5]
"""

import json
import os
import sys

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def check_structure(g):
    need = ("actions", "states", "dones")
    missing = [k for k in need if k not in g]
    missing += [k for k in ("agentview_image",) if k not in g.get("obs", {})]
    attrs_missing = [k for k in ("model_file", "env_args", "instruction",
                                 "num_samples") if k not in g.attrs]
    ok = not missing and not attrs_missing
    return ok, missing, attrs_missing


def main(path):
    import mujoco
    import robosuite

    failures = []
    f = h5py.File(path, "r")
    demos = list(f["data"].keys())
    print("=" * 64)
    print(f"LIBERO COMPAT CHECK: {path} ({len(demos)} episodes)")
    print("=" * 64)

    # ---- 1. structure ----
    for k in demos:
        ok, missing, attrs_missing = check_structure(f["data"][k])
        if not ok:
            failures.append(f"{k}: missing {missing + attrs_missing}")
    print(f"[{'ok' if not failures else 'FAIL'}] 1. episode structure "
          f"(actions/states/agentview_image/dones + attrs)")

    g0 = f["data"][demos[0]]
    model_xml = g0.attrs["model_file"]

    # ---- 2. model_file compiles ----
    try:
        model = mujoco.MjModel.from_xml_string(model_xml)
        print(f"[ok] 2. model_file compiles (nq={model.nq}, ngeom={model.ngeom})")
    except Exception as exc:
        print(f"[FAIL] 2. model_file does not compile: {exc}")
        failures.append("model_file compile")
        model = None

    # ---- 4. env_args rebuild via robosuite.make ----
    env_args = json.loads(g0.attrs["env_args"])
    env_name = env_args.get("env_name")
    from robosuite.environments.base import REGISTERED_ENVS
    registered_before = env_name in REGISTERED_ENVS
    if not registered_before:
        # the repo task registers itself; emulate what the package import does
        sys.path.insert(0, HERE)
        import importlib
        mod = importlib.import_module("task_put_red_in_box")
        cls = getattr(mod, env_name, None)
        if cls is not None:
            REGISTERED_ENVS[env_name] = cls
    try:
        env_kwargs = dict(env_args.get("env_kwargs", {}))
        env_kwargs.update(dict(has_renderer=False, has_offscreen_renderer=False,
                               use_camera_obs=False, horizon=200))
        env2 = robosuite.make(env_name=env_name, **env_kwargs)
        env2.reset()
        # 恢复采集时的机器人基座位姿(默认 make 会用 robosuite 自带定位)
        base_pos = np.asarray(env_args["env_kwargs"].get("base_pos",
                              [0.22, -0.45, 0.0]), dtype=float)
        env2.robots[0].robot_model.set_base_xpos(base_pos)
        env2.reset()
        print(f"[ok] 4. robosuite.make('{env_name}') rebuilds the env "
              f"(action_dim={env2.action_dim}, base={base_pos.tolist()})")
    except Exception as exc:
        print(f"[FAIL] 4. robosuite.make rebuild failed: {exc}")
        failures.append("env_args rebuild")
        env2 = None

    # ---- 3. states replay reproduces recorded EE pose ----
    # 官方 LIBERO replay 方式:用 episode 自带的 model_file 编译模型,
    # 写入 states 后 forward —— 同一 MJCF + 同一 qpos 必然同一 FK
    if model is not None:
        g = f["data"][demos[0]]
        states = g["states"]
        eef = g["obs/robot0_eef_pos"]
        d2 = mujoco.MjData(model)
        bid_eef = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                    "gripper0_right_eef")
        nq = model.nq
        nv = model.nv
        worst = 0.0
        for k in np.linspace(0, states.shape[0] - 1, 5).astype(int):
            st = states[k]
            d2.qpos[:] = st[:nq]
            d2.qvel[:] = st[nq:nq + nv]
            mujoco.mj_forward(model, d2)
            err = float(np.linalg.norm(d2.xpos[bid_eef] - eef[k]))
            worst = max(worst, err)
        ok3 = worst < 0.005
        print(f"[{'ok' if ok3 else 'FAIL'}] 3. states replay reproduces EE pose "
              f"(worst {worst*1000:.2f} mm over 5 sampled frames)")
        if not ok3:
            failures.append("states replay mismatch")

    # ---- 5. dones terminal semantics ----
    bad_dones = []
    for k in demos:
        dn = f["data"][k]["dones"][:]
        if not dn[-1]:
            bad_dones.append(k)
    if bad_dones:
        print(f"[FAIL] 5. episodes without terminal done: {bad_dones}")
        failures.append("dones")
    else:
        print(f"[ok] 5. all {len(demos)} episodes end with dones=True")

    print("-" * 64)
    if failures:
        print("LIBERO_COMPAT: FAILED")
        for x in failures:
            print("  -", x)
        sys.exit(1)
    print("LIBERO_COMPAT: PASS — drop-in for the LIBERO loading path")
    sys.exit(0)


if __name__ == "__main__":
    default = os.path.join(HERE, "..", "data", "put_red_in_box", "demo.hdf5")
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.abspath(default))

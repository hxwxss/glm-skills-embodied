# -*- coding: utf-8 -*-
"""
run_pipeline.py — Agentic Scene-to-Data 管线一键端到端运行
==========================================================

  [1] IR 编译 MuJoCo              compile_mjcf.py         → generated/lab_scene.xml
  [3] 静态物理自检                test_mjcf_physics.py    → settle/穿透/静止
  [4] 可达性预检                  check_reachability.py   → 全部物体在 Panda 工作域
  [5] robosuite 任务自检          task_put_red_in_box.py  → reset 分布/初始判据
  [6] IK 专家验收                 expert_ik.py test       → 成功率 ≥80%
  [7] 演示数据采集                collect_demos.py        → data/demo.hdf5

用法:
    python run_pipeline.py                # 全流程
    python run_pipeline.py --episodes 4   # 缩小采集规模(调试)
退出码 0 = 全部阶段通过。
"""

import argparse
import subprocess
import sys
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PYTHON = sys.executable


def run(name, cmd, check_keywords=()):
    print("\n" + "=" * 66)
    print("STAGE:", name)
    print("=" * 66)
    r = subprocess.run(cmd, cwd=HERE)
    out = r.returncode == 0
    if out and check_keywords:
        pass  # 关键字校验在子脚本内部以退出码表达
    print(f"===> {name}: {'PASS' if out else 'FAIL'}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--collect-episodes", type=int, default=6)
    args = ap.parse_args()

    ok = True
    stages = [
        ("1. MJCF compile",
         [PYTHON, os.path.join(HERE, "compile_mjcf.py")]),
        ("2. Physics self-check",
         [PYTHON, os.path.join(HERE, "test_mjcf_physics.py")]),
        ("3. Reachability pre-check",
         [PYTHON, os.path.join(HERE, "check_reachability.py")]),
        ("4. robosuite task self-check",
         [PYTHON, os.path.join(HERE, "task_put_red_in_box.py"),
          "--reset-test", "10"]),
        ("5. IK expert acceptance (>=80%)",
         [PYTHON, os.path.join(HERE, "expert_ik.py"), "test",
          "--episodes", str(args.episodes)]),
        ("6. Penetration audit",
         [PYTHON, os.path.join(HERE, "test_penetration.py")]),
        ("7. Demo collection",
         [PYTHON, os.path.join(HERE, "collect_demos.py"),
          "--episodes", str(args.collect_episodes),
          "--out", os.path.join(HERE, "..", "data", "demo.hdf5")]),
    ]
    for name, cmd in stages:
        ok &= run(name, cmd)

    print("\n" + "=" * 66)
    print("PIPELINE_RESULT:", "ALL_OK" if ok else "FAILED")
    print("=" * 66)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

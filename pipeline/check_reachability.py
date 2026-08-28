# -*- coding: utf-8 -*-
"""Scene-spec reachability pre-check for task-critical objects."""
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
with (HERE / "spec" / "scene_spec.json").open(encoding="utf-8") as fh:
    spec = json.load(fh)
REACH_R = 0.78          # Panda 实用工作半径(留操作余量),标称 0.855
MIN_R = 0.15            # 太近会撞底盘

robot = next(o for o in spec["robots"])
bx, by, bz = robot["base_pos"]
fail = []
# 仅校验任务必须交互的对象(抓取目标/容器);静态装饰(干扰球/面板/灯)
# 不需要可达,只需不与机器人初始位形碰撞
MUST_REACH = {"grasp_target", "container"}
for obj in spec["objects"]:
    if not (set(obj.get("semantic", [])) & MUST_REACH):
        continue
    p = obj.get("body_pos") or obj["pos"]
    d = math.hypot(p[0]-bx, p[1]-by)
    ok = MIN_R < d <= REACH_R
    status = "ok" if ok else "FAIL"
    print(f"[{status}] {obj['id']:22s} horizontal reach = {d:.3f} m "
          f"(allowed {MIN_R}-{REACH_R})")
    if not ok:
        fail.append(obj["id"])

if fail:
    print("REACHABILITY_FAILED:", fail)
    sys.exit(1)
print("REACHABILITY_OK")

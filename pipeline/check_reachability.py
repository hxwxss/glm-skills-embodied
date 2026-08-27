# -*- coding: utf-8 -*-
"""spec.yaml 达性预检:桌面任务物体的水平距离须在 Panda 工作半径内."""
import json, math, sys

spec = json.load(open(r"..\spec\scene_spec.json", encoding="utf-8"))
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

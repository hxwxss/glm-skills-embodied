# -*- coding: utf-8 -*-
"""
render_views.py — 按固定相机渲染工具
用法:
  blender --background embodied_lab.blend --factory-startup \
      --python render_views.py -- --cams overview,tabletop,interaction_closeup \
      --out renders/iteration_01 --samples 64 [--res 1280x720] [--tag ]

相机名映射:
  overview            -> Cam_Overview
  tabletop            -> Cam_Tabletop
  interaction_closeup -> Cam_InteractionCloseup
"""
import bpy
import os
import sys

CAM_MAP = {
    "overview": "Cam_Overview",
    "tabletop": "Cam_Tabletop",
    "interaction_closeup": "Cam_InteractionCloseup",
}


def parse_args():
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    opts = {"cams": ["overview", "tabletop", "interaction_closeup"],
            "out": "renders/iteration_00", "samples": None,
            "res": None, "tag": ""}
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--cams":
            opts["cams"] = argv[i + 1].split(","); i += 2
        elif a == "--out":
            opts["out"] = argv[i + 1]; i += 2
        elif a == "--samples":
            opts["samples"] = int(argv[i + 1]); i += 2
        elif a == "--res":
            opts["res"] = argv[i + 1]; i += 2
        elif a == "--tag":
            opts["tag"] = argv[i + 1]; i += 2
        else:
            i += 1
    return opts


def main():
    opts = parse_args()
    sc = bpy.context.scene
    # 注意：Blender 渲染输出必须用绝对路径，相对路径在后台模式下会静默写丢
    out_dir = os.path.abspath(opts["out"])
    os.makedirs(out_dir, exist_ok=True)
    if opts["samples"]:
        sc.cycles.samples = int(opts["samples"])
    if opts["res"]:
        w, h = opts["res"].lower().split("x")
        sc.render.resolution_x = int(w)
        sc.render.resolution_y = int(h)

    for tag in opts["cams"]:
        cam_name = CAM_MAP.get(tag)
        cam = bpy.data.objects.get(cam_name) if cam_name else None
        if not cam:
            cam = bpy.data.objects.get(tag)
        if cam is None or cam.type != 'CAMERA':
            print(f"RENDER_SKIP no camera for '{tag}' (looked up {cam_name})")
            continue
        sc.camera = cam
        frame = tag if not opts["tag"] else f"{tag}_{opts['tag']}"
        out_path = os.path.join(out_dir, f"{frame}.png")
        sc.render.filepath = out_path.replace("\\", "/")
        print(f"RENDER_START {frame} engine={sc.render.engine} device="
              f"{getattr(sc.cycles, 'device', '?')} samples={sc.cycles.samples} "
              f"cam={cam.name} loc=({cam.location.x:.2f},{cam.location.y:.2f},{cam.location.z:.2f})")
        bpy.ops.render.render(write_still=True)
        size = os.path.getsize(out_path) if os.path.exists(out_path) else -1
        print(f"RENDER_DONE {frame} -> {out_path} bytes={size}")
    print("RENDER_ALL_OK")


if __name__ == "__main__":
    main()

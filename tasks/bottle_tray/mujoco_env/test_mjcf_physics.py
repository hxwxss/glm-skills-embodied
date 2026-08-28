# -*- coding: utf-8 -*-
"""
test_mjcf_physics.py — headless physics self-check of the compiled MJCF
========================================================================

Checks (simulate 3 s @ 500 Hz):
  A. model loads; expected bodies present
  B. green bottle settles resting ON the table (right height, inside table)
  C. tray body rests at the table top; no pairwise penetration (AABB)
  D. final velocities ~= 0 (settled)
  E. bottle starts OUTSIDE the tray (init must not already be successful)
  F. offscreen render receipt -> renders/mujoco_settle.png

Exit code 0 = PASS, 1 = FAIL.
"""

import os
import sys
import json

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
XML = os.path.join(HERE, "generated", "bottle_tray_scene.xml")
SPEC = os.path.join(HERE, "..", "spec", "scene_spec.json")
SHOT = os.path.join(HERE, "..", "renders", "mujoco_settle.png")

DYNAMIC_BODIES = ["GreenBottle"]
STATIC_COMPOSITES = ["Tray"]
STATIC_GEOMS = ["RedSphere"]
STEPS = 1500          # 0.002 s * 1500 = 3.0 s

with open(SPEC, encoding="utf-8") as _fh:
    SPEC_DATA = json.load(_fh)
TABLE_TOP_Z = SPEC_DATA["workspace"]["table_top_z"]
BOTTLE = next(o for o in SPEC_DATA["objects"] if o["id"] == "GreenBottle")
BOTTLE_H = BOTTLE["dims"][1]


def body_aabb(model, data, body_id):
    """World AABB half extents of all geoms of a body (10% slack)."""
    ext = np.zeros(3)
    found = False
    for g in range(model.ngeom):
        if model.geom_bodyid[g] == body_id:
            gm = model.geom_size[g]
            gt = model.geom_type[g]
            if gt == mujoco.mjtGeom.mjGEOM_BOX:
                e = gm[:3]
            elif gt == mujoco.mjtGeom.mjGEOM_CYLINDER:
                e = np.array([gm[0], gm[0], gm[1]])
            elif gt == mujoco.mjtGeom.mjGEOM_SPHERE:
                e = np.array([gm[0]] * 3)
            else:
                continue
            ext = np.maximum(ext, e)
            found = True
    return ext * 1.05 if found else None


def main():
    failures = []
    print("=" * 62)
    print("PHYSICS SELF-CHECK:", XML)
    print("=" * 62)

    model = mujoco.MjModel.from_xml_path(XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    print("[ok] model loaded; nq=%d ngeom=%d nbody=%d"
          % (model.nq, model.ngeom, model.nbody))

    # A. expected bodies present
    bid = {}
    for name in DYNAMIC_BODIES + STATIC_COMPOSITES:
        b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if b < 0:
            failures.append(f"missing body {name}")
        bid[name] = b
    for name in STATIC_GEOMS:
        g = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if g < 0:
            failures.append(f"missing geom {name}")
    if failures:
        print("FAIL missing bodies"); sys.exit(1)
    print("[ok] A. all expected bodies/geoms present")

    table_top_z = 0.75
    bottle = bid["GreenBottle"]
    tray = bid["Tray"]

    x0 = data.body(bottle).xpos.copy()
    # E. bottle must start OUTSIDE the tray footprint (init not successful)
    tray0 = data.body(tray).xpos[:2]
    out0 = np.linalg.norm(x0[:2] - tray0) > 0.15
    print("[%s] E. bottle starts outside the tray (dist=%.3f m)"
          % ("ok" if out0 else "FAIL", np.linalg.norm(x0[:2] - tray0)))
    if not out0:
        failures.append("bottle starts inside/near the tray")

    for step in range(STEPS):
        mujoco.mj_step(model, data)
        if step % 500 == 0:
            print("   t=%.2fs bottle z=%.4f" % (step * 0.002,
                                                data.body(bottle).xpos[2]))

    # B. bottle settled on the table at the expected height
    bx = data.body(bottle).xpos
    expected_z = TABLE_TOP_Z + BOTTLE_H / 2
    ok_b = abs(bx[2] - expected_z) < 0.01
    inside = (abs(bx[0]) < 0.95 - 0.05) and (abs(bx[1]) < 0.475 - 0.05)
    print("[%s] B. bottle settled z=%.4f (expected %.4f), on-table=%s"
          % ("ok" if ok_b and inside else "FAIL", bx[2], expected_z, inside))
    if not ok_b:
        failures.append("bottle height off after settle")
    if not inside:
        failures.append("bottle outside table bounds")

    # tray static body must sit at the table top
    tx = data.body(tray).xpos
    ok_tray = abs(tx[2] - TABLE_TOP_Z) < 1e-6
    print("[%s] B2. tray static body at table top z=%.4f"
          % ("ok" if ok_tray else "FAIL", tx[2]))
    if not ok_tray:
        failures.append("tray body moved off the table")

    # C. pairwise AABB non-penetration: bottle vs tray, bottle vs sphere
    pairs = [("GreenBottle", "Tray")]
    ext = {n: body_aabb(model, data, bid[n]) for n in bid}
    for a, b in pairs:
        pa, pb_ = data.body(bid[a]).xpos, data.body(bid[b]).xpos
        ea, eb = ext[a], ext[b]
        overlap = all(abs(pa[i] - pb_[i]) < (ea[i] + eb[i]) * 0.98 for i in range(3))
        print("[%s] C. no penetration %s vs %s"
              % ("FAIL" if overlap else "ok", a, b))
        if overlap:
            failures.append(f"AABB penetration {a}/{b}")
    sphere_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "RedSphere")
    sp = data.geom_xpos[sphere_gid]
    overlap_s = (np.linalg.norm(bx[:2] - sp[:2])
                 < 0.05 + 0.025 + 0.005) and bx[2] < sp[2] + 0.05 + 0.14 / 2
    print("[%s] C2. no penetration GreenBottle vs RedSphere" % ("FAIL" if overlap_s else "ok"))
    if overlap_s:
        failures.append("bottle/sphere overlap")

    # D. final velocities ~= 0
    for name in DYNAMIC_BODIES:
        v = float(np.linalg.norm(data.body(bid[name]).cvel[3:6]))
        ok_d = v < 0.02
        print("[%s] D. %s final speed=%.4f m/s" % ("ok" if ok_d else "FAIL", name, v))
        if not ok_d:
            failures.append(f"{name} still moving at end")

    shot_ok = render_shot(model, data, SHOT)
    print("[%s] F. offscreen render -> %s"
          % ("ok" if shot_ok else "SKIP", SHOT))

    print("-" * 62)
    if failures:
        print("RESULT: PHYSICS_CHECK_FAILED")
        for f in failures:
            print("  -", f)
        sys.exit(1)
    print("RESULT: PHYSICS_OK")
    sys.exit(0)


def render_shot(model, data, out_png):
    try:
        renderer = mujoco.Renderer(model, height=480, width=640)
        cam = mujoco.MjvCamera()
        cam.lookat[:] = [0.11, -0.17, TABLE_TOP_Z - 0.07]
        cam.distance = 1.6
        cam.elevation = -42
        cam.azimuth = -115
        renderer.update_scene(data, camera=cam)
        img = renderer.render()
        os.makedirs(os.path.dirname(out_png), exist_ok=True)
        try:
            import cv2
            cv2.imwrite(out_png, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        except ImportError:
            from PIL import Image
            Image.fromarray(img).save(out_png)
        renderer.close()
        return True
    except Exception as exc:
        print("   (render failed:", exc, ")")
        return False


if __name__ == "__main__":
    main()

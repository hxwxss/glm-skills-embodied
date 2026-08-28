# -*- coding: utf-8 -*-
"""
test_mjcf_physics.py -- M1 gate: headless physics self-check of tier_scene.xml
==============================================================================

Checks (3 s settle @ 500 Hz plus dedicated articulation probes):
  A. model loads; the lid is a REAL hinge joint and the drawer a REAL slide
     joint (named, expected type/range, world-anchored)
  B. closed-state stability: lid rests closed, drawer closed, both cubes rest
     on the table at the expected height; all final speeds ~ 0; no pairwise
     penetration
  C. lid flip-and-hold: a transient impulse opens the lid; with zero further
     actuation it settles PAST VERTICAL at the open stop and stays
  D. drawer slide-and-hold: a transient impulse pulls the drawer out along
     -y; the horizontal slide has no gravity back-drive, so it stays where
     it stops (drift < 2 mm over the last second)
  E. offscreen render receipts: closed state + open state

Exit code 0 = PASS, 1 = FAIL.
"""

import argparse
import os
import sys

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
XML = os.path.join(HERE, "generated", "tier_scene.xml")
RENDER_DIR = os.path.normpath(os.path.join(HERE, "..", "renders"))

EXPECTED_HINGE_RANGE = (-2.1, 0.0)
EXPECTED_SLIDE_RANGE = (0.0, 0.150)
LID_BODY = "Lid"
DRAWER_BODY = "Drawer"
HINGE_JOINT = "lid_hinge"
SLIDE_JOINT = "drawer_slide"
TABLE_TOP_Z = 0.75
CUBE_HALF = 0.021


def run_steps(model, data, n, act=None):
    for _ in range(n):
        if act is not None:
            act(data)
        mujoco.mj_step(model, data)


def max_penetration(data):
    worst = 0.0
    for i in range(data.ncon):
        c = data.contact[i]
        if c.dist < 0:
            worst = max(worst, -c.dist)
    return worst


def body_speed(model, data):
    speeds = [float(np.linalg.norm(data.body(b).cvel[3:6]))
              for b in range(1, model.nbody)]
    return max(speeds)


def render(model, data, cam_name, out_png):
    try:
        renderer = mujoco.Renderer(model, height=480, width=640)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        cam.fixedcamid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA,
                                           cam_name)
        opt = mujoco.MjvOption()
        opt.geomgroup[0] = 0
        renderer.update_scene(data, camera=cam, scene_option=opt)
        img = renderer.render()
        renderer.close()
        os.makedirs(os.path.dirname(out_png), exist_ok=True)
        from PIL import Image
        Image.fromarray(img).save(out_png)
        return True
    except Exception as exc:
        print("   (render failed: %s)" % exc)
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", default=XML)
    args = ap.parse_args()

    failures = []
    print("=" * 64)
    print("PHYSICS SELF-CHECK:", args.xml)
    print("=" * 64)

    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    print("[ok] model loaded; nq=%d ngeom=%d nbody=%d"
          % (model.nq, model.ngeom, model.nbody))

    # ---- A. real articulations ------------------------------------------
    hid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, HINGE_JOINT)
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, SLIDE_JOINT)
    if hid < 0 or sid < 0:
        print("[FAIL] A. joints not found (hinge=%d slide=%d)" % (hid, sid))
        sys.exit(1)
    hrng = model.jnt_range[hid]
    srng = model.jnt_range[sid]
    lid_ok = (model.jnt_type[hid] == mujoco.mjtJoint.mjJNT_HINGE and
              abs(hrng[0] - EXPECTED_HINGE_RANGE[0]) < 1e-6 and
              abs(hrng[1] - EXPECTED_HINGE_RANGE[1]) < 1e-6 and
              model.body_parentid[model.jnt_bodyid[hid]] == 0)
    dr_ok = (model.jnt_type[sid] == mujoco.mjtJoint.mjJNT_SLIDE and
             abs(srng[0] - EXPECTED_SLIDE_RANGE[0]) < 1e-6 and
             abs(srng[1] - EXPECTED_SLIDE_RANGE[1]) < 1e-6 and
             model.body_parentid[model.jnt_bodyid[sid]] == 0)
    print("[%s] A. lid: real hinge range=%s parent=world=%s"
          % ("ok" if lid_ok else "FAIL", np.round(hrng, 3).tolist(),
             model.body_parentid[model.jnt_bodyid[hid]] == 0))
    print("[%s] A. drawer: real slide range=%s parent=world=%s"
          % ("ok" if dr_ok else "FAIL", np.round(srng, 3).tolist(),
             model.body_parentid[model.jnt_bodyid[sid]] == 0))
    if not lid_ok:
        failures.append("lid hinge check")
    if not dr_ok:
        failures.append("drawer slide check")

    h_adr = model.jnt_qposadr[hid]
    s_adr = model.jnt_qposadr[sid]
    h_dof = model.jnt_dofadr[hid]
    s_dof = model.jnt_dofadr[sid]

    # ---- B. closed-state settle ------------------------------------------
    run_steps(model, data, 1500)                     # 3.0 s settle
    theta = float(data.qpos[h_adr])
    slide = float(data.qpos[s_adr])
    ok_b = abs(theta) <= 0.03 and abs(slide) <= 0.002
    print("[%s] B. rests closed: lid theta=%.4f rad, drawer slide=%.4f m"
          % ("ok" if ok_b else "FAIL", theta, slide))
    if not ok_b:
        failures.append("closed state not stable")

    max_speed = body_speed(model, data)
    ok_speed = max_speed < 0.02
    print("[%s] B. all bodies near rest (max |v| = %.4f m/s)"
          % ("ok" if ok_speed else "FAIL", max_speed))
    if not ok_speed:
        failures.append("bodies still moving after settle")

    pen = max_penetration(data)
    ok_pen = pen <= 0.008
    print("[%s] B. max contact penetration %.2f mm (<= 8 mm)"
          % ("ok" if ok_pen else "FAIL", pen * 1000))
    if not ok_pen:
        failures.append("penetration %.1f mm" % (pen * 1000))

    for cid, expect_z in (("RedCube", TABLE_TOP_Z + CUBE_HALF),
                          ("BlueCube", TABLE_TOP_Z + CUBE_HALF)):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, cid)
        z = float(data.body(bid).xpos[2])
        okz = abs(z - expect_z) < 0.005
        print("[%s] B. %s rests at z=%.4f (expected ~%.4f)"
              % ("ok" if okz else "FAIL", cid, z, expect_z))
        if not okz:
            failures.append("%s not resting on table" % cid)

    render(model, data, "agentview",
           os.path.join(RENDER_DIR, "m1_closed_agentview.png"))
    render(model, data, "sideview",
           os.path.join(RENDER_DIR, "m1_closed_sideview.png"))

    # ---- C. lid flip-and-hold --------------------------------------------
    def push_lid(d):
        d.qvel[h_adr] = -1.5

    run_steps(model, data, 800, push_lid)            # kinematic push 1.6 s
    run_steps(model, data, 1500)                     # passive dynamics 3 s
    theta_open = float(data.qpos[h_adr])
    tail = []
    for _ in range(500):
        mujoco.mj_step(model, data)
        tail.append(float(data.qpos[h_adr]))
    wander = abs(tail[-1] - tail[0])
    speed_open = float(abs(data.qvel[h_adr]))
    past_vertical = theta_open <= -1.62
    at_stop = theta_open >= EXPECTED_HINGE_RANGE[0] - 0.06
    holds = wander < 0.005 and speed_open < 0.02
    ok_c = past_vertical and at_stop and holds
    print("[%s] C. flip-and-hold: theta=%.3f rad (%.0f deg) stop=%s "
          "1s wander=%.4f speed=%.4f"
          % ("ok" if ok_c else "FAIL", theta_open, np.degrees(theta_open),
             at_stop, wander, speed_open))
    if not ok_c:
        failures.append("lid does not flip past vertical and stay open")

    render(model, data, "agentview",
           os.path.join(RENDER_DIR, "m1_open_agentview.png"))
    render(model, data, "sideview",
           os.path.join(RENDER_DIR, "m1_open_sideview.png"))

    # ---- D. drawer slide-and-hold ----------------------------------------
    def pull_drawer(d):
        d.qfrc_applied[s_dof] = 3.0    # steady pull toward the robot

    run_steps(model, data, 600, pull_drawer)         # pull for 1.2 s
    data.qfrc_applied[s_dof] = 0.0
    run_steps(model, data, 1500)                     # coast / stop 3 s
    slide_open = float(data.qpos[s_adr])
    tail = []
    for _ in range(500):
        mujoco.mj_step(model, data)
        tail.append(float(data.qpos[s_adr]))
    drift = abs(tail[-1] - tail[0])
    speed_dr = float(abs(data.qvel[s_adr]))
    pulled = slide_open >= 0.08
    stays = drift < 0.002 and speed_dr < 0.01
    ok_d = pulled and stays
    print("[%s] D. slide-and-hold: slide=%.3f m (>=0.08), 1s drift=%.1f mm, "
          "speed=%.4f"
          % ("ok" if ok_d else "FAIL", slide_open, drift * 1000, speed_dr))
    if not ok_d:
        failures.append("drawer does not stay pulled out")
    pen_d = max_penetration(data)
    print("[%s] D. penetration after articulations: %.2f mm"
          % ("ok" if pen_d <= 0.008 else "FAIL", pen_d * 1000))
    if pen_d > 0.008:
        failures.append("penetration after articulation %.1f mm"
                        % (pen_d * 1000))

    print("-" * 64)
    if failures:
        print("RESULT: PHYSICS_CHECK_FAILED")
        for f in failures:
            print("  -", f)
        sys.exit(1)
    print("RESULT: PHYSICS_OK")
    sys.exit(0)


if __name__ == "__main__":
    main()

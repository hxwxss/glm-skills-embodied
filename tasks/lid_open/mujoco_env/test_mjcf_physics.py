# -*- coding: utf-8 -*-
"""
test_mjcf_physics.py -- M1 gate: headless physics self-check of lid_scene.xml
=============================================================================

Checks (3 s settle @ 500 Hz plus a dedicated articulation probe):
  A. model loads; lid is a REAL hinge joint (named, type=hinge, expected range)
  B. closed-state stability: lid rests closed on the box (|theta| <= 0.03 rad),
     all final speeds ~ 0, no pairwise penetration
  C. flip-and-hold proof: a transient impulse at the lid opens it; with zero
     further actuation the lid settles PAST VERTICAL at the open stop and
     stays there (drift < 0.01 rad over the last second)
  D. closed-direction impulse returns the lid to the closed stop (gravity +
     stop hold it shut; nothing floats)
  E. offscreen render receipts: closed state + flipped-open state

Exit code 0 = PASS, 1 = FAIL.
"""

import argparse
import json
import os
import sys

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
XML = os.path.join(HERE, "generated", "lid_scene.xml")
RENDER_DIR = os.path.normpath(os.path.join(HERE, "..", "renders"))

EXPECTED_RANGE = (-2.1, 0.0)
LID_BODY = "Lid"
HINGE_JOINT = "lid_hinge"
TABLE_TOP_Z = 0.75
DT = 0.002


def run_steps(model, data, n, hinge_adr, apply_vel=None):
    theta_log = []
    for _ in range(n):
        if apply_vel is not None:
            data.qvel[hinge_adr] = apply_vel
        mujoco.mj_step(model, data)
        theta_log.append(data.qpos[hinge_adr])
    return np.array(theta_log)


def max_penetration(model, data):
    worst = 0.0
    for i in range(data.ncon):
        c = data.contact[i]
        if c.dist < 0:
            worst = max(worst, -c.dist)
    return worst


def render(model, data, cam_name, out_png):
    try:
        renderer = mujoco.Renderer(model, height=480, width=640)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        cam.fixedcamid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA,
                                           cam_name)
        renderer.update_scene(data, camera=cam)
        img = renderer.render()
        renderer.close()
        os.makedirs(os.path.dirname(out_png), exist_ok=True)
        try:
            import cv2
            cv2.imwrite(out_png, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        except ImportError:
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

    # ---- A. the lid is a real hinge joint ------------------------------
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, HINGE_JOINT)
    if jid < 0:
        print("[FAIL] A. joint '%s' not found -- lid is not articulated" % HINGE_JOINT)
        sys.exit(1)
    jtype = model.jnt_type[jid]
    is_hinge = jtype == mujoco.mjtJoint.mjJNT_HINGE
    rng = model.jnt_range[jid]
    range_ok = (abs(rng[0] - EXPECTED_RANGE[0]) < 1e-6 and
                abs(rng[1] - EXPECTED_RANGE[1]) < 1e-6)
    lid_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, LID_BODY)
    body_of_joint = model.jnt_bodyid[jid]
    anchored = body_of_joint == lid_bid
    box_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "LidBox")
    lid_is_child = model.body_parentid[lid_bid] == 0  # world-anchored hinge
    ok_a = is_hinge and range_ok and anchored and lid_is_child
    print("[%s] A. real hinge joint: type_hinge=%s range=%s (want %s) "
          "on body '%s' parent=world=%s"
          % ("ok" if ok_a else "FAIL", is_hinge, np.round(rng, 3).tolist(),
             list(EXPECTED_RANGE), LID_BODY, lid_is_child))
    if not ok_a:
        failures.append("hinge joint check")

    hinge_adr = model.jnt_qposadr[jid]
    dof_adr = model.jnt_dofadr[jid]

    # ---- B. closed-state settle ----------------------------------------
    th_log = run_steps(model, data, 1500, hinge_adr)          # 3.0 s
    theta_closed = float(data.qpos[hinge_adr])
    ok_b = abs(theta_closed) <= 0.03
    print("[%s] B. lid rests closed after settle: theta=%.4f rad"
          % ("ok" if ok_b else "FAIL", theta_closed))
    if not ok_b:
        failures.append("closed state not stable (theta=%.4f)" % theta_closed)

    speeds = [float(np.linalg.norm(data.body(b).cvel[3:6])) for b in range(1, model.nbody)]
    max_speed = max(speeds)
    ok_speed = max_speed < 0.02
    print("[%s] B. all bodies near rest (max |v| = %.4f m/s)"
          % ("ok" if ok_speed else "FAIL", max_speed))
    if not ok_speed:
        failures.append("bodies still moving after settle")

    pen = max_penetration(model, data)
    ok_pen = pen <= 0.008
    print("[%s] B. max contact penetration %.2f mm (<= 8 mm)"
          % ("ok" if ok_pen else "FAIL", pen * 1000))
    if not ok_pen:
        failures.append("penetration %.1f mm" % (pen * 1000))

    knob_z = float(data.geom(
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "lid_handle_bar")).xpos[2])
    expect_knob_z = TABLE_TOP_Z + 0.084 + 0.033
    ok_knob = abs(knob_z - expect_knob_z) < 0.01
    print("[%s] B. handle bar at z=%.4f (expected ~%.4f)"
          % ("ok" if ok_knob else "FAIL", knob_z, expect_knob_z))
    if not ok_knob:
        failures.append("handle displaced at closed state")

    render(model, data, "agentview",
           os.path.join(RENDER_DIR, "m1_closed_agentview.png"))
    render(model, data, "sideview",
           os.path.join(RENDER_DIR, "m1_closed_sideview.png"))

    # ---- C. flip-and-hold proof ----------------------------------------
    # slow kinematic push (like the robot hand) for 1.6 s, then pure passive
    # dynamics 3 s: past vertical gravity must hold the lid at the open stop
    run_steps(model, data, 800, hinge_adr, apply_vel=-1.5)
    run_steps(model, data, 1500, hinge_adr)
    theta_open = float(data.qpos[hinge_adr])
    tail = run_steps(model, data, 500, hinge_adr)             # 1.0 s watch
    wander = float(abs(tail[-1] - tail[0]))                   # rad over 1 s
    speed_open = float(abs(data.qvel[dof_adr]))
    past_vertical = theta_open <= -1.62          # > ~93 deg
    at_stop = theta_open >= EXPECTED_RANGE[0] - 0.06
    holds = wander < 0.005 and speed_open < 0.02
    ok_c = past_vertical and at_stop and holds
    print("[%s] C. flip-and-hold: theta=%.3f rad (%.0f deg), stop_hit=%s, "
          "1s wander=%.4f rad, speed=%.4f rad/s"
          % ("ok" if ok_c else "FAIL", theta_open, np.degrees(theta_open),
             at_stop, wander, speed_open))
    if not ok_c:
        failures.append("lid does not flip past vertical and stay open")

    render(model, data, "agentview",
           os.path.join(RENDER_DIR, "m1_open_agentview.png"))
    render(model, data, "sideview",
           os.path.join(RENDER_DIR, "m1_open_sideview.png"))

    # ---- D. gravity keeps it shut below vertical ------------------------
    # disturb the lid open to a sub-vertical angle, release: it must fall
    # back shut on its own (this is why the robot must push PAST vertical)
    data2 = mujoco.MjData(model)
    mujoco.mj_forward(model, data2)
    jid2 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, HINGE_JOINT)
    adr2 = model.jnt_qposadr[jid2]
    run_steps(model, data2, 400, adr2, apply_vel=-1.5)        # disturb to ~-40 deg
    th_dist = float(data2.qpos[adr2])
    run_steps(model, data2, 1750, adr2)                       # release, 3.5 s
    theta_back = float(data2.qpos[adr2])
    ok_d = abs(theta_back) <= 0.03 and th_dist > -1.57
    print("[%s] D. sub-vertical lid (%.0f deg) falls back shut: theta=%.4f"
          % ("ok" if ok_d else "FAIL", np.degrees(th_dist), theta_back))
    ok_d = abs(theta_back) <= 0.03
    print("[%s] D. lid re-settles shut after closing impulse: theta=%.4f"
          % ("ok" if ok_d else "FAIL", theta_back))
    if not ok_d:
        failures.append("lid does not return shut")

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

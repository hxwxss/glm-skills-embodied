# -*- coding: utf-8 -*-
"""
build_scene.py -- M0: scene authoring -> Intermediate Representation (IR)
=========================================================================

Single source of truth for the "Panda opens a hinged-lid box" tabletop scene.
All layout / geometry / mass / task constants live here; the MuJoCo backends
(standalone MJCF compiler and the robosuite arena) only *compile* this IR and
never hard-code geometry.

Scene (all z relative to floor):
  * lab table 1.9 x 0.95 m, top surface at z = 0.75
  * Panda on a RethinkMount pedestal at the table edge, facing +y
  * small storage box (0.16 x 0.12 x 0.078 m) on the table
  * hinged lid: real MuJoCo hinge joint anchored at the box back-top edge,
    range [-2.1, 0] rad. theta = 0 is closed; negative theta lifts the front
    edge. Past -90 deg gravity holds the lid against the open stop, so an
    opened lid stays open with zero actuation.
  * grasp knob (stem + sphere head) near the lid's front edge: a top-down,
    yaw-invariant grasp target for the parallel gripper.

Usage:
    python build_scene.py [--out spec/scene_spec.json]
"""

import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# workspace
TABLE_TOP_Z = 0.75
TABLE_SIZE_XY = (1.9, 0.95)
TABLE_HALF_THICK = 0.0225
TABLE_RGBA = (0.20, 0.21, 0.23, 1.0)
FLOOR_Z = 0.0

# ---------------------------------------------------------------------------
# robot (Panda on tall pedestal at the table edge, facing +y)
ROBOT = {
    "id": "Panda_0",
    "type": "panda",
    "mount": "RethinkMount",          # raises shoulder to tabletop height
    "base_pos": [0.22, -0.45, 0.0],   # table front edge is at y = -0.475
    "base_yaw_deg": 0.0,
}

# ---------------------------------------------------------------------------
# hinged-lid storage box ("LidBox").  body origin sits on the table top,
# at the box center.  Local frame: x = long axis, y = depth (+y = back,
# where the hinge is; the robot stands at -y).
BOX_CENTER_XY = (0.30, 0.08)
BOX_OUTER = (0.16, 0.12)       # x, y outer footprint
BOTTOM_THICK = 0.008
WALL_THICK = 0.008
WALL_HEIGHT = 0.070            # walls rise from top of bottom to z = 0.078
BOX_RGBA = (0.85, 0.45, 0.10, 0.55)      # translucent amber
BOTTOM_RGBA = (0.45, 0.24, 0.06, 1.0)
RIM_RGBA = (1.0, 0.62, 0.0, 1.0)         # solid orange rim on the top edges

# hinged lid, expressed in the box body frame
LID_THICK = 0.012
LID_PANEL_HALF = (BOX_OUTER[0] / 2 + 0.001,     # 1 mm overhang each side
                  BOX_OUTER[1] / 2 + 0.001,
                  LID_THICK / 2)
RIM_TOP_Z = BOTTOM_THICK + WALL_HEIGHT                   # 0.078
LID_HINGE_POS = (0.0, BOX_OUTER[1] / 2 + 0.002,          # 2 mm behind back face
                 RIM_TOP_Z + LID_THICK / 2)              # lid mid-plane
HINGE_AXIS = (1.0, 0.0, 0.0)
HINGE_RANGE_RAD = (-2.1, 0.0)    # closed stop at 0, open stop at -2.1
HINGE_DAMPING = 0.02             # realistic free-swing hinge; gravity alone
                                 # must close it below vertical and hold it
                                 # open past vertical
LID_MASS = 0.10
LID_RGBA = (0.88, 0.89, 0.92, 1.0)   # near-white: high contrast when open

# grasp handle on the lid: a cylindrical BAR parallel to the hinge axis,
# on a bracket at the lid's front edge.  The Panda's pads press its +-x
# flanks along the full bar length (line contact) -- a sphere knob pops out
# of the grip mid-pull, a trapped bar does not.  Local coords in the
# *lid/hinge* body frame; lid local -y points toward the front edge.
BAR_R = 0.009
BAR_HALF_LEN = 0.020            # 4 cm bar along y (parallel to hinge)
BAR_Y = -0.102                  # near the lid front edge
BAR_Z = LID_THICK / 2 + 0.027   # bar center 2.7 cm above the lid top
BRACKET_SIZE = (0.030, 0.010, 0.009)
BRACKET_POS = (0.0, BAR_Y, LID_THICK / 2 + BRACKET_SIZE[2])
HANDLE_RGBA = (0.13, 0.14, 0.16, 1.0)
# lid front edge trimmed to 11 mm beyond the bar: any lid material further
# forward sweeps up into the palm mid-pull and pries the grip open
LID_FRONT_Y = -0.113
LID_BACK_Y = 0.002

# ---------------------------------------------------------------------------
# static decorations (far from the task zone; scene dressing only)
DECORATIONS = [
    {
        "id": "Control_Panel_Deck", "physics": "static", "shape": "box",
        "dims": [0.20, 0.13, 0.022], "pos": [-0.38, -0.30, TABLE_TOP_Z + 0.022],
        "rot_euler_xyz": [-0.1745, 0.0, 0.1396],
        "rgba": [0.10, 0.11, 0.125, 1.0], "semantic": ["obstacle"],
    },
    {
        "id": "Desk_Lamp_Base", "physics": "static", "shape": "cylinder",
        "dims": [0.085, 0.024], "pos": [-0.78, 0.26, TABLE_TOP_Z + 0.024],
        "rgba": [0.55, 0.56, 0.58, 1.0], "semantic": ["obstacle"],
    },
]

# ---------------------------------------------------------------------------
# cameras (pos + look-at target, compiled to MuJoCo xyaxes form)
CAMERAS = [
    {"id": "agentview", "pos": [0.31, 0.46, 1.30],
     "target_xyz": [0.28, 0.06, 0.78], "fov_deg": 45},
    {"id": "sideview", "pos": [1.15, -0.32, 1.12],
     "target_xyz": [0.28, 0.08, 0.80], "fov_deg": 40},
]

# ---------------------------------------------------------------------------
# task
TASK = {
    "name": "OpenBoxLid",
    "instruction": "flip open the lid of the storage box and leave it open",
    "type": "articulated_open",
    "target_articulation": "lid_hinge",
    "grasp_target_geom": "lid_handle_bar",
    "success_condition": {
        "lid_angle_max_rad": -1.75,     # lid at least ~100 deg open
        "lid_max_speed_rad_s": 0.35,    # and (near) stationary when checked
    },
    "joint_limit_rad": list(HINGE_RANGE_RAD),
    "expert": {
        "hover_height_m": 0.12,
        "grasp_dz_m": 0.0,              # EE target = handle center + dz
                                        # (calibrated: both pads pinch the
                                        #  bar, no lid contact)
        "arc_start_rad": -0.35,
        "arc_end_rad": -1.92,           # ~ -110 deg, well past vertical
        "arc_steps": 10,
        "arc_dwell": 8,                 # let the lid catch up between waypoints
        "prerelease_lift_m": 0.05,      # lift clear of the raised lid BEFORE
                                        # opening the fingers (otherwise the
                                        # hand knocks the lid back shut)
        "close_dwell": 45,
        "hold_dwell": 30,
        "release_dwell": 30,
        "settle_dwell": 45,
        "retreat_offset_m": [0.0, -0.10, 0.13],
    },
    "init_randomization": {
        "box_xy_jitter_m": 0.025,       # whole box+lid assembly, per episode
    },
}


def build_spec():
    cx, cy = BOX_CENTER_XY
    wx, wy = BOX_OUTER[0] / 2, BOX_OUTER[1] / 2
    t = WALL_THICK
    walls = [
        {"name": "front", "pos": [0.0, -wy + t / 2, BOTTOM_THICK + WALL_HEIGHT / 2],
         "size": [wx, t / 2, WALL_HEIGHT / 2]},
        {"name": "back", "pos": [0.0, wy - t / 2, BOTTOM_THICK + WALL_HEIGHT / 2],
         "size": [wx, t / 2, WALL_HEIGHT / 2]},
        {"name": "left", "pos": [-wx + t / 2, 0.0, BOTTOM_THICK + WALL_HEIGHT / 2],
         "size": [t / 2, wy - t, WALL_HEIGHT / 2]},
        {"name": "right", "pos": [wx - t / 2, 0.0, BOTTOM_THICK + WALL_HEIGHT / 2],
         "size": [t / 2, wy - t, WALL_HEIGHT / 2]},
        {"name": "bottom", "pos": [0.0, 0.0, BOTTOM_THICK / 2],
         "size": [wx, wy, BOTTOM_THICK / 2], "role": "bottom"},
    ]

    lid_box = {
        "id": "LidBox",
        "physics": "static_composite",
        "semantic": ["container", "fixture"],
        "body_pos": [cx, cy, TABLE_TOP_Z],
        "walls": walls,
        "rgba": list(BOX_RGBA),
        "bottom_rgba": list(BOTTOM_RGBA),
        "rim_rgba": list(RIM_RGBA),
        "hinged_lid": {
            "id": "Lid",
            "joint": "lid_hinge",
            "hinge_body_pos": [cx + LID_HINGE_POS[0], cy + LID_HINGE_POS[1],
                               TABLE_TOP_Z + LID_HINGE_POS[2]],
            "axis": list(HINGE_AXIS),
            "range_rad": list(HINGE_RANGE_RAD),
            "damping": HINGE_DAMPING,
            "geoms": [
                {"name": "lid_panel", "shape": "box",
                 "pos": [0.0, (LID_FRONT_Y + LID_BACK_Y) / 2, 0.0],
                 "size": [LID_PANEL_HALF[0],
                          (LID_BACK_Y - LID_FRONT_Y) / 2,
                          LID_PANEL_HALF[2]],
                 "mass": LID_MASS,
                 "rgba": list(LID_RGBA), "friction": [0.6, 0.01, 0.0005]},
                {"name": "lid_handle_bracket", "shape": "box",
                 "pos": list(BRACKET_POS), "size": list(BRACKET_SIZE),
                 "mass": 0.012, "rgba": list(HANDLE_RGBA),
                 "friction": [1.0, 0.01, 0.0005]},
                {"name": "lid_handle_bar", "shape": "cylinder",
                 "pos": [0.0, BAR_Y, BAR_Z],
                 "size": [BAR_R, BAR_HALF_LEN],
                 "quat_euler_xyz_deg": [90.0, 0.0, 0.0],   # axis -> world y
                 "mass": 0.018, "rgba": list(HANDLE_RGBA),
                 "friction": [2.0, 0.01, 0.0005],   # rubbery bar: the pad
                                                    # grip must hold the pull
                 "semantic": ["grasp_target"]},
            ],
            "knob_local": [0.0, BAR_Y, BAR_Z],
        },
    }

    spec = {
        "schema": "scene-spec/v0.3-lid",
        "scene_name": "panda_lid_open_tabletop",
        "units": "meters",
        "up_axis": "z",
        "gravity": [0.0, 0.0, -9.81],
        "workspace": {
            "table_top_z": TABLE_TOP_Z,
            "table_size_xy": list(TABLE_SIZE_XY),
            "table_half_thickness": TABLE_HALF_THICK,
            "table_rgba": list(TABLE_RGBA),
            "floor_z": FLOOR_Z,
        },
        "objects": DECORATIONS + [lid_box],
        "robots": [ROBOT],
        "cameras": CAMERAS,
        "task": TASK,
        "provenance": {
            "source_script": "build_scene.py",
            "generated_by": "GLM-5.3-Flash coding agent",
        },
    }
    return spec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "spec", "scene_spec.json"))
    args = ap.parse_args()

    spec = build_spec()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(spec, fh, indent=2)

    lid = spec["objects"][-1]["hinged_lid"]
    print("[m0] IR written ->", args.out)
    print("[m0] box body_pos =", spec["objects"][-1]["body_pos"])
    print("[m0] hinge anchor =", lid["hinge_body_pos"],
          "range =", lid["range_rad"], "damping =", lid["damping"])
    print("[m0] knob (lid-local) =", lid["knob_local"])
    return args.out


if __name__ == "__main__":
    main()

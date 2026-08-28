# -*- coding: utf-8 -*-
"""
build_scene.py -- M0: scene authoring -> Intermediate Representation (IR)
=========================================================================

Single source of truth for the "Panda sorts two cubes into a two-tier
storage box" tabletop scene.  All layout / geometry / mass / task constants
live here; the MuJoCo backends (standalone MJCF compiler and the robosuite
arena) only *compile* this IR and never hard-code geometry.

Scene (all z relative to the floor):
  * lab table 1.9 x 0.95 m, top surface at z = 0.75
  * Panda on a RethinkMount pedestal at the table edge, facing +y
  * two-tier storage box (0.22 x 0.15 m footprint) on the table:
      - lower tier: a real pull-out DRAWER (MuJoCo slide joint, range
        [0, 0.115] m along -y).  The slide is horizontal, so gravity has no
        component along it: wherever the arm leaves the drawer, it stays.
      - upper tier: closed compartment with a real HINGED LID (MuJoCo hinge
        joint at the box back-top edge, range [-2.1, 0] rad).  Past ~ -90 deg
        gravity holds the lid against the open stop: opened lid stays open
        with zero actuation.
  * both tiers expose a cylindrical grasp bar (top-down grasp, pads pinch
    the bar's +-x flanks -- the reference-proven grip)
  * red cube (4.2 cm, fits the measured ~5.9 cm PandaGripper aperture)
    and blue cube (4.2 cm) on the table

Task: open the lid, pull the drawer out, put the red cube in the upper
compartment, put the blue cube in the drawer.

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
# two-tier storage box ("TierBox").  body origin sits on the table top, at
# the box center.  Local frame: x = long axis, y = depth (+y = back, where
# the hinge is; -y = front, where the drawer mouth is; the robot stands at
# -y).  All child geoms are specified in this box frame.
BOX_CENTER_XY = (0.30, 0.10)
BOX_OUTER = (0.22, 0.15)          # x, y outer footprint
T = 0.008                         # universal wall / plate thickness
LOWER_H = 0.085                   # lower tier total height (0 -> 0.085)
UPPER_WALL_H = 0.080              # upper walls: LOWER_H -> LOWER_H + 0.080
RIM_TOP_Z = LOWER_H + UPPER_WALL_H                       # 0.165
BOX_RGBA = (0.30, 0.34, 0.42, 1.0)        # slate-blue housing
TOP_RGBA = (0.22, 0.25, 0.32, 1.0)        # compartment floor (top plate)
RIM_RGBA = (1.0, 0.62, 0.0, 1.0)          # solid orange rim strips

# hinged lid, expressed in the LID/HINGE body frame (origin at the hinge)
LID_THICK = 0.012
LID_HINGE_POS = (0.0, BOX_OUTER[1] / 2 + 0.002,   # 2 mm behind back face
                 RIM_TOP_Z + LID_THICK / 2)       # lid mid-plane
HINGE_AXIS = (1.0, 0.0, 0.0)
HINGE_RANGE_RAD = (-2.1, 0.0)     # closed stop at 0, open stop at -2.1
HINGE_DAMPING = 0.02
LID_MASS = 0.14
LID_RGBA = (0.88, 0.89, 0.92, 1.0)   # near-white: high contrast when open
# lid panel: covers the box top, front edge trimmed ~9 mm short of the box
# front so the panel never sweeps up into the palm mid-pull (pitfall)
LID_FRONT_LOCAL_Y = -0.143        # = -(0.075 + 0.068); box front -0.075
LID_BACK_LOCAL_Y = 0.002
# grasp bars on the lid front edge and drawer front: SQUARE section so the
# pads cannot roll them out of the grip under a sustained pull/tangential
# arc (a round bar creeps and pops out; the square bar keeps full-face
# contact).  Same envelope as the reference-proven round bar.
BAR_R = 0.009                     # bar half-section (square 18 x 18 mm)
BAR_HALF_LEN = 0.020              # 4 cm bar along y
LID_BAR_LOCAL = (0.0, -0.133, LID_THICK / 2 + 0.027)
BRACKET_SIZE = (0.015, 0.005, 0.0045)     # MJCF half-extents
LID_BRACKET_LOCAL = (0.0, -0.133, LID_THICK / 2 + 0.009)
HANDLE_RGBA = (0.13, 0.14, 0.16, 1.0)

# drawer ("Drawer" body, child of the box body, slide joint along -y)
DRAWER_BODY_POS = (0.0, -0.013, T)        # closed: centered x, resting on
                                          # the housing bottom plate top
SLIDE_AXIS = (0.0, -1.0, 0.0)             # +qpos pulls toward the robot
SLIDE_RANGE_M = (0.0, 0.150)              # long stroke: the open tray cavity
                                          # ends up ~9 cm clear of the box
                                          # face, so the wrist column never
                                          # leans on the upper walls when
                                          # lowering into the tray
SLIDE_DAMPING = 0.35
SLIDE_FRICTIONLOSS = 0.40                 # N; horizontal slide: no gravity
                                          # back-drive, stays where left
TRAY_W = 0.098                    # tray wall half sizes (MJCF half-extents)
TRAY_L = 0.070                    # cavity 12.4 cm in y: the gripper palm
                                  # (~10 cm long) must fit between the walls
                                  # at release depth, palm hanging below the
                                  # wall tops
TRAY_WALL_H = 0.016               # half height -> walls 0.006..0.038
TRAY_BOTTOM_T = 0.006
TRAY_RGBA = (0.55, 0.38, 0.16, 1.0)       # warm wood-tone tray
# drawer handle: chest-handle style -- the bar protrudes 6.5 cm in front of
# the tray wall so the WHOLE hand column (measured: +-5 cm around the EE
# origin in y, 3.3 cm below it) clears the housing top plate lip when
# grasping.  Measured palm front reaches eef_y + 0.051; bar center at box
# y = -0.138 puts the palm front ~1 cm clear of the plate edge (box
# y = -0.075).  The bracket is the protruding shank.
DRAWER_BAR_LOCAL = (0.0, -0.125, 0.0605)  # in the drawer body frame
DRAWER_BRACKET_LOCAL = (0.0, -0.0925, 0.0425)
DRAWER_BRACKET_SIZE = (0.015, 0.033, 0.0045)

# ---------------------------------------------------------------------------
# graspable props (free bodies; compiled to BoxObjects in the robosuite task)
CUBE_HALF = 0.021                  # 4.2 cm cube
CUBE_MASS = 0.05
CUBE_FRICTION = [2.0, 0.008, 0.0002]   # rubbery: the pad grip must hold the
                                       # cube against shear during carries
PROPS = [
    {
        "id": "RedCube", "physics": "free", "shape": "box",
        "dims": [2 * CUBE_HALF] * 3, "mass": CUBE_MASS,
        "rgba": [0.85, 0.15, 0.12, 1.0], "semantic": ["grasp_target"],
        "friction": list(CUBE_FRICTION),
        "pos": [-0.02, -0.16, TABLE_TOP_Z + CUBE_HALF],
        "task_goal": "upper_compartment",
    },
    {
        "id": "BlueCube", "physics": "free", "shape": "box",
        "dims": [2 * CUBE_HALF] * 3, "mass": CUBE_MASS,
        "rgba": [0.15, 0.35, 0.85, 1.0], "semantic": ["grasp_target"],
        "friction": list(CUBE_FRICTION),
        "pos": [0.50, -0.18, TABLE_TOP_Z + CUBE_HALF],
        "task_goal": "drawer",
    },
]

# ---------------------------------------------------------------------------
# static decorations (far from the task zone; scene dressing only)
DECORATIONS = [
    {
        "id": "Control_Panel_Deck", "physics": "static", "shape": "box",
        "dims": [0.20, 0.13, 0.022], "pos": [-0.55, -0.30, TABLE_TOP_Z + 0.022],
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
    # front view from the robot side: the box, drawer pull and both cubes
    # are all in front of the camera (a +y-side camera looks at the box
    # back and the open lid blocks the task)
    {"id": "agentview", "pos": [0.02, -0.62, 1.22],
     "target_xyz": [0.34, 0.08, 0.72], "fov_deg": 50},
    {"id": "sideview", "pos": [1.15, -0.35, 1.12],
     "target_xyz": [0.30, 0.02, 0.80], "fov_deg": 40},
]

# ---------------------------------------------------------------------------
# task
TASK = {
    "name": "TwoTierSort",
    "instruction": ("open the lid, pull out the drawer, put the red cube in "
                    "the upper compartment and the blue cube in the drawer, "
                    "then close the drawer and the lid again"),
    "type": "articulated_open_place_close",
    "success_condition": {
        # zones are computed by build_spec() from the geometry above
        "max_speed_m_s": 0.05,
        "lid_closed_rad": 0.06,       # |hinge angle| at the end
        "drawer_closed_m": 0.012,     # |slide distance| at the end
    },
    "init_randomization": {
        "box_xy_jitter_m": 0.02,        # whole box+lid+drawer assembly
        "prop_xy_jitter_m": 0.04,       # each cube
        # yaw kept small: the parallel-ripper pads must square up to the cube
        # faces (the expert aligns to the live yaw; beyond ~7 deg the flat
        # pads would meet the faces at an edge, degrading the grasp)
        "prop_yaw_jitter_rad": 0.12,
    },
    "expert": {
        "lid": {
            "hover_height_m": 0.12,
            "grasp_dz_m": 0.0,
            "arc_start_rad": -0.35,
            "arc_end_rad": -1.92,       # ~ -110 deg, well past vertical
            "arc_steps": 14,            # fine waypoints: the bar lags the
                                        # joint targets mid-arc, small jumps
                                        # keep the pad grip from shearing off
            "arc_dwell": 14,
            "prerelease_lift_m": 0.05,
            "close_dwell": 45,
            "hold_dwell": 30,
            "release_dwell": 30,
            "retreat_offset_m": [0.0, -0.06, 0.14],
        },
        "drawer": {
            "hover_height_m": 0.10,
            "grasp_dz_m": 0.0,
            "pull_distance_m": 0.150,
            "pull_steps": 12,
            "pull_dwell": 8,
            "close_dwell": 45,
            "release_dwell": 20,
            "retreat_up_m": 0.12,
            "retreat_offset_m": [0.0, -0.06, 0.10],
        },
        "close_drawer": {
            "hover_height_m": 0.10,
            "grasp_dz_m": 0.0,
            "push_steps": 8,
            "push_dwell": 4,
            "push_overshoot_m": 0.008,  # target past the closed stop; the
                                        # joint limit seats the tray flush
            "close_dwell": 45,
            "release_dwell": 20,
            "retreat_up_m": 0.12,
            "retreat_offset_m": [0.0, -0.08, 0.10],
        },
        "close_lid": {
            "hover_height_m": 0.10,
            "grasp_dz_m": 0.0,
            "arc_end_rad": -0.15,       # release just short of shut; gravity
                                        # closes the last ~9 deg gently
            "arc_steps": 10,
            "arc_dwell": 8,
            "close_dwell": 45,
            "release_dwell": 25,
            "retreat_up_m": 0.13,
            "retreat_offset_m": [0.0, -0.08, 0.12],
        },
        "pick": {
            "hover_height_m": 0.11,
            "grasp_dz_m": 0.008,        # EE target slightly above cube
                                        # center: with ~5 mm gravity droop
                                        # the pads land on the cube's upper
                                        # half, clear of the table
            "close_dwell": 45,
            "carry_z_m": 1.00,          # absolute transit height (8.5 cm
                                        # above the box rim)
            "release_drop_m": 0.042,    # EE rest height above the zone floor
                                        # = drop + grasp clearance (small
                                        # drop -> no bounce-out)
            "release_dwell": 25,
            "retreat_up_m": 0.11,
        },
        "settle_dwell": 90,
        "final_retreat_offset_m": [0.0, -0.16, 0.12],
    },
}


def _housing_walls():
    """Static geoms of the lower-tier housing + upper compartment walls
    (box-local).  The housing is a nightstand shell: bottom plate, side
    walls, back wall, top plate -- the front face stays open (drawer mouth)."""
    hx, hy = BOX_OUTER[0] / 2, BOX_OUTER[1] / 2
    zc = (T + LOWER_H) / 2              # side/back wall center z
    zh = (LOWER_H - T) / 2
    up_zc = LOWER_H + UPPER_WALL_H / 2
    uy = hy - T                         # upper side walls y half size
    return [
        # lower housing
        {"name": "housing_bottom", "pos": [0.0, 0.0, T / 2],
         "size": [hx, hy, T / 2], "role": "floor"},
        {"name": "housing_left", "pos": [-hx + T / 2, 0.0, zc],
         "size": [T / 2, hy, zh]},
        {"name": "housing_right", "pos": [hx - T / 2, 0.0, zc],
         "size": [T / 2, hy, zh]},
        {"name": "housing_back", "pos": [0.0, hy - T / 2, zc],
         "size": [hx, T / 2, zh]},
        {"name": "housing_top", "pos": [0.0, 0.0, LOWER_H - T / 2],
         "size": [hx, hy, T / 2], "role": "compartment_floor"},
        # upper compartment
        {"name": "upper_front", "pos": [0.0, -hy + T / 2, up_zc],
         "size": [hx, T / 2, UPPER_WALL_H / 2]},
        {"name": "upper_back", "pos": [0.0, hy - T / 2, up_zc],
         "size": [hx, T / 2, UPPER_WALL_H / 2]},
        {"name": "upper_left", "pos": [-hx + T / 2, 0.0, up_zc],
         "size": [T / 2, uy, UPPER_WALL_H / 2]},
        {"name": "upper_right", "pos": [hx - T / 2, 0.0, up_zc],
         "size": [T / 2, uy, UPPER_WALL_H / 2]},
    ]


def build_spec():
    cx, cy = BOX_CENTER_XY

    lid = {
        "id": "Lid",
        "joint": "lid_hinge",
        "hinge_body_pos": [cx + LID_HINGE_POS[0], cy + LID_HINGE_POS[1],
                           TABLE_TOP_Z + LID_HINGE_POS[2]],
        "axis": list(HINGE_AXIS),
        "range_rad": list(HINGE_RANGE_RAD),
        "damping": HINGE_DAMPING,
        "geoms": [
            {"name": "lid_panel", "shape": "box",
             "pos": [0.0, (LID_FRONT_LOCAL_Y + LID_BACK_LOCAL_Y) / 2, 0.0],
             "size": [BOX_OUTER[0] / 2 + 0.001,
                      (LID_BACK_LOCAL_Y - LID_FRONT_LOCAL_Y) / 2,
                      LID_THICK / 2],
             "mass": LID_MASS, "rgba": list(LID_RGBA),
             "friction": [0.6, 0.01, 0.0005]},
            {"name": "lid_handle_bracket", "shape": "box",
             "pos": list(LID_BRACKET_LOCAL), "size": list(BRACKET_SIZE),
             "mass": 0.012, "rgba": list(HANDLE_RGBA),
             "friction": [1.0, 0.01, 0.0005]},
            {"name": "lid_handle_bar", "shape": "box",
             "pos": list(LID_BAR_LOCAL),
             "size": [BAR_R, BAR_HALF_LEN, BAR_R],
             "mass": 0.018, "rgba": list(HANDLE_RGBA),
             "friction": [2.0, 0.01, 0.0005],   # rubbery bar: the pad grip
             "semantic": ["grasp_target"]},     # must hold the pull
        ],
        "grasp_local": list(LID_BAR_LOCAL),
    }

    tray_t = 0.008
    wcz = TRAY_BOTTOM_T + TRAY_WALL_H        # tray wall center z = 0.022
    drawer = {
        "id": "Drawer",
        "joint": "drawer_slide",
        "body_pos": [cx + DRAWER_BODY_POS[0], cy + DRAWER_BODY_POS[1],
                     TABLE_TOP_Z + DRAWER_BODY_POS[2]],
        "axis": list(SLIDE_AXIS),
        "range_m": list(SLIDE_RANGE_M),
        "damping": SLIDE_DAMPING,
        "frictionloss": SLIDE_FRICTIONLOSS,
        "geoms": [
            {"name": "tray_bottom", "shape": "box",
             "pos": [0.0, 0.0, TRAY_BOTTOM_T / 2],
             "size": [TRAY_W, TRAY_L, TRAY_BOTTOM_T / 2], "mass": 0.14,
             "rgba": list(TRAY_RGBA), "friction": [0.5, 0.008, 0.0002]},
            {"name": "tray_front", "shape": "box",
             "pos": [0.0, -TRAY_L + tray_t / 2, wcz],
             "size": [TRAY_W, tray_t / 2, TRAY_WALL_H], "mass": 0.03,
             "rgba": list(TRAY_RGBA)},
            {"name": "tray_back", "shape": "box",
             "pos": [0.0, TRAY_L - tray_t / 2, wcz],
             "size": [TRAY_W, tray_t / 2, TRAY_WALL_H], "mass": 0.03,
             "rgba": list(TRAY_RGBA)},
            {"name": "tray_left", "shape": "box",
             "pos": [-TRAY_W + tray_t / 2, 0.0, wcz],
             "size": [tray_t / 2, TRAY_L - tray_t, TRAY_WALL_H], "mass": 0.03,
             "rgba": list(TRAY_RGBA)},
            {"name": "tray_right", "shape": "box",
             "pos": [TRAY_W - tray_t / 2, 0.0, wcz],
             "size": [tray_t / 2, TRAY_L - tray_t, TRAY_WALL_H], "mass": 0.03,
             "rgba": list(TRAY_RGBA)},
            {"name": "drawer_handle_bracket", "shape": "box",
             "pos": list(DRAWER_BRACKET_LOCAL),
             "size": list(DRAWER_BRACKET_SIZE), "mass": 0.012,
             "rgba": list(HANDLE_RGBA), "friction": [1.0, 0.01, 0.0005]},
            {"name": "drawer_handle_bar", "shape": "box",
             "pos": list(DRAWER_BAR_LOCAL),
             "size": [BAR_R, BAR_HALF_LEN, BAR_R],
             "mass": 0.018, "rgba": list(HANDLE_RGBA),
             "friction": [2.0, 0.01, 0.0005],
             "semantic": ["grasp_target"]},
        ],
        "grasp_local": list(DRAWER_BAR_LOCAL),
    }

    tier_box = {
        "id": "TierBox",
        "physics": "static_composite",
        "semantic": ["container", "fixture"],
        "body_pos": [cx, cy, TABLE_TOP_Z],
        "walls": _housing_walls(),
        "rgba": list(BOX_RGBA),
        "floor_rgba": list(TOP_RGBA),
        "rim_rgba": list(RIM_RGBA),
        "hinged_lid": lid,
        "drawer": drawer,
        # upper compartment cavity, box-local (success zone for the red cube);
        # half = cavity inner half - cube half: a cube resting flush against
        # any wall still counts as inside
        "upper_zone_local": {
            "center": [0.0, 0.0, LOWER_H + CUBE_HALF],
            "half": [BOX_OUTER[0] / 2 - T - CUBE_HALF,
                     BOX_OUTER[1] / 2 - T - CUBE_HALF,
                     UPPER_WALL_H / 2],
        },
        # drawer cavity, drawer-local (success zone for the blue cube)
        "tray_zone_local": {
            "center": [0.0, 0.0, TRAY_BOTTOM_T + CUBE_HALF],
            "half": [TRAY_W - tray_t - CUBE_HALF,
                     TRAY_L - tray_t - CUBE_HALF,
                     TRAY_WALL_H],
        },
    }

    spec = {
        "schema": "scene-spec/v0.4-tier",
        "scene_name": "panda_two_tier_sort",
        "units": "meters",
        "up_axis": "z",
        "gravity": [0.0, 0.0, -9.81],
        "workspace": {
            "table_top_z": TABLE_TOP_Z,
            "table_size_xy": list(TABLE_SIZE_XY),
            "table_half_thickness": TABLE_HALF_THICK,
            "table_rgba": [0.20, 0.21, 0.23, 1.0],
            "floor_z": FLOOR_Z,
        },
        "objects": DECORATIONS + [tier_box] + PROPS,
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
    ap.add_argument("--out", default=os.path.join(HERE, "spec",
                                                  "scene_spec.json"))
    args = ap.parse_args()

    spec = build_spec()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(spec, fh, indent=2)

    box = spec["objects"][len(DECORATIONS)]
    lid, drawer = box["hinged_lid"], box["drawer"]
    print("[m0] IR written ->", args.out)
    print("[m0] box body_pos =", box["body_pos"])
    print("[m0] lid hinge =", lid["hinge_body_pos"],
          "range =", lid["range_rad"])
    print("[m0] drawer body_pos =", drawer["body_pos"],
          "axis =", drawer["axis"], "range =", drawer["range_m"])
    print("[m0] lid grasp local =", lid["grasp_local"],
          " drawer grasp local =", drawer["grasp_local"])
    for p in spec["objects"]:
        if p.get("physics") == "free":
            print("[m0] prop %-8s at %s -> %s"
                  % (p["id"], p["pos"], p["task_goal"]))
    return args.out


if __name__ == "__main__":
    main()

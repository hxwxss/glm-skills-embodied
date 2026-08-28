# Two-Tier Box Sort — Panda + MuJoCo (agentic-sim2data pipeline)

MuJoCo tabletop scene: a Panda arm at the edge of a lab table faces a
**two-tier storage box** with a real openable lid (upper level) and a real
pull-out drawer (lower level), plus a red and a blue cube.  The arm
**opens the box** (flips the lid past vertical, pulls the drawer out) and
sorts: **red cube → upper compartment, blue cube → drawer**.

## Artifacts

| artifact | path |
| --- | --- |
| one successful demonstration (video) | `rollouts/two_tier_demo.mp4` (side-by-side agentview+sideview, 640×480 @20 fps) |
| LIBERO/robosuite-style dataset | `demos/demo.hdf5` (readback-verified) |
| scene IR (single source of truth) | `spec/scene_spec.json` |
| compiled standalone MJCF | `mujoco_env/generated/tier_scene.xml` |
| render receipts / debug frames | `renders/` |

## Pipeline stages and gates (all passing)

| stage | gate | result |
| --- | --- | --- |
| M0 scene build → IR | IR written | OK |
| M1 IR → MJCF + physics self-check | settle, no pairwise penetration, lid flip-and-hold, drawer slide-and-hold, render receipts | PHYSICS_OK |
| M1.5 reachability pre-check | every waypoint IK-solvable, radius 0.15–0.78 m | REACHABILITY_OK (62 waypoints) |
| M2 robosuite task | N random resets: success stays False at init, jitter bounds respected, articulations shut | OK |
| M3 expert acceptance | success ≥ 80% over ≥ 6 randomized episodes | 5/6 = 83% (last full run; verified-phase policy — see iteration log) |
| M3.5 penetration audit | zero arm-link collisions, no pair > 8 mm sustained / 25 mm spike, placements survive a 1.5 s settle | PASS |
| M4 demo capture | HDF5 written, every episode success=True, shapes verified by reading the file back, MP4 rendered | READBACK OK |

One command (runs every gate in order, exit 0 = all passed):

```
cd mujoco_env && python run_pipeline.py --episodes 6
```

## Scene design (all numbers live in `build_scene.py`)

* table 1.9 × 0.95 m, top at z = 0.75; Panda on a RethinkMount at
  (0.22, −0.45) facing +y.
* **Two-tier box** (0.22 × 0.15 m footprint) at (0.30, 0.10):
  * lower tier: nightstand-style housing (bottom plate / sides / back /
    top plate, **open front**); a wooden tray rides on a real MuJoCo
    **slide joint** (axis −y, range 0–0.15 m, frictionloss 0.4 N).  The
    stroke is horizontal, so gravity cannot back-drive it: the drawer
    stays wherever the arm leaves it.  The handle is a chest-handle bar
    protruding 6.5 cm in front of the tray wall — far enough that the
    whole hand column clears the housing plate lip when grasping.
  * upper tier: closed compartment (walls up to z = 0.165 box-local) with
    a real **hinge joint** at the back-top edge (range −2.1…0 rad).  Past
    ~−90° gravity holds the lid against the open stop: opened stays open.
  * both tiers expose an 18 mm **square-section grasp bar** (square, so
    the pads cannot roll it out of the grip during a sustained pull or
    the lid arc) with the reference-proven top-down pad grip.
* **cubes** 4.2 cm (inside the measured ~5.9 cm PandaGripper aperture),
  rubbery friction (μ = 2) so the pad grip survives carry shear; red at
  (−0.02, −0.16), blue at (0.50, −0.18).
* success = red cube center inside the upper-compartment cavity AND blue
  cube inside the live drawer-cavity zone, both near-stationary.

## Expert (M3)

mink numerical IK + absolute JOINT_POSITION control (kp 300, input limits
±4.5 rad — robosuite silently clamps to ±1 otherwise).  Per episode, in
order: flip lid open along the exact hinge arc → pull drawer to its open
stop → pick red, place in the upper compartment → pick blue, place in the
drawer → settle.  Key structural choices that the gates forced:

1. **Verified phases with retries** — each phase (lid / drawer / pick-red /
   pick-blue) has a geometric verifier; a failed phase re-runs from the
   live state (max 3 attempts).  This lifted the raw policy from ~50% to
   8/8, absorbing the residual flakiness of real contact-rich grasping.
2. **Lazy waypoints** — pick/place keypoints resolve at execution time
   (the tray zone only exists after the pull; a carried cube moves with
   the hand).
3. **Yaw-aligned, frozen grasp orientation** — the grasp rotation is
   Rz(live cube yaw)·top-down, captured once per phase: square to the
   faces (a ±34° spawn yaw presents the corner diagonal to the pads) and
   immune to the carried cube's pendulum sway.
4. **Seeded-only IK while carrying** — no wrist-branch fallback mid-carry
   (a branch switch sweeps the joints and flings the grasp).
5. **ensure_open / recovery_lift** — the pick phases pop a jammed finger
   (pulse close-then-open) and climb to carry height before a retry
   descent (found by the M3.5 audit: a blind retry pressed the hand into
   the cube, 9 mm).

## Dataset schema (`demos/demo.hdf5`)

```
attrs: spec_snapshot (full IR), success_rate, instruction
data/demo_i
  attrs: num_samples, success, instruction
  actions   (T, 8)      absolute joint targets (7) + gripper (1)
  obs/
    agentview_image (T, 256, 256, 3)   mujoco.Renderer, live model
    robot0_eef_pos (T, 3)   robot0_eef_quat (T, 4)
    robot0_joint_pos (T, 7) robot0_gripper_qpos (T, 2)
    lid_qpos (T,)  drawer_qpos (T,)
    red_cube_pos (T, 3)     blue_cube_pos (T, 3)
  dones     (T,)
  env_args: env_name, spec_json
```

## Iteration history

See `iteration_log.md` — every gate failure and the fix it forced.

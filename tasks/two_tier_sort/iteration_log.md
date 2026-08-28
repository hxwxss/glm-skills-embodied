# Iteration Log — two_tier_box

Every gate failure and the design fix it forced.  Fixes are applied in the
IR/builder (`build_scene.py`) and the whole pipeline reruns — never patched
in the backend only.

## Round 1 — M1 physics self-check

1. **Serialization crash**: the dynamic-geom emitter built
   `ET.Element("geom", {...})` with raw float attributes (mass 0.14) —
   ElementTree cannot serialize floats.  Fix: route through the shared
   `sub()` formatter.
2. **Drawer slide-and-hold FAIL (slide stayed 0.000)**: the test pushed the
   drawer by force-setting `qvel` every step — an infinitely stiff
   prescription that overrides the soft joint-limit constraint, so the
   drawer overshot its range and the state was garbage.  Fix: apply a
   bounded `qfrc_applied` (3 N) instead.  With a real force the drawer
   slides to the stop (0.115 m) and stays (speed ≈ 0, zero drift).
3. **0.5 mm persistent penetration `drawer_handle_bar ↔ housing_top`**: the
   bar's half-length was forgotten when checking clearances — its rear end
   poked into the housing top plate's front lip.  Fix: move the bar
   forward (box y −0.080 → −0.088) and re-derive bracket geometry.
   Penetration at rest dropped to 0.03 mm.

## Round 2 — M1.5 reachability

4. **Carry height bug**: `carry_z` was treated as table-relative
   (0.75 + 0.97 = 1.72 m) — the wrist strained vertically, IK rot error
   27–55°.  Fix: absolute carry z = 1.00 m.
5. **Gate/solver tolerance mismatch**: IK broke at 6 mm while the gate
   demanded 5 mm (one arc waypoint stopped at 5.2 mm).  Fix: solver break
   tolerance 3 mm.  All 52→62 waypoints then solve ≤ 2 mm.

## Round 3 — M2 task wrapper

6. **Missing `obj_body_id`**: `prop_pos` crashed (free-joint body ids are
   registered in `_setup_references`).  Fix: map name → `root_body` id
   per reset (hard_reset rebuilds the model every episode).
7. **Missing `import sys`** in the reset-test path (exit code gate).

## Round 4 — M3 expert (the long one)

8. **Stale plan-time poses**: ALL keypoints were computed at episode start,
   but the drawer's tray zone only exists AFTER the pull — the blue cube
   was lowered onto the closed-tray position (under the housing top
   plate), the descent stalled 400 steps at the plate, and the cube was
   released on top of the housing.  Fix: pick-phase waypoints are LAZY
   (resolved from live state at execution time).
9. **Hand column vs housing plate**: measured (not guessed) gripper
   extents at the drawer grasp pose: the hand spans ±5 cm around the EE
   origin in y and 3.3 cm below it — the palm rammed the housing top
   plate's front edge and the tray wall.  Fix: chest-handle style bar
   protruding 6.5 cm in front of the tray wall.
10. **Tray cavity narrower than the palm**: at release depth the palm
    hangs below the wall tops and is ~10 cm long in y vs a 10.4 cm
    cavity → palm clips the tray front wall.  Fix: TRAY_L 0.060 → 0.070
    (12.4 cm cavity).
11. **Cube slip during carry**: telemetry showed the grip spread going
    83.7 → 3.5 mm (empty) between transit and lower — the cube slid out
    of the smooth pads.  Fix: rubbery cube friction (μ = 2) in the IR
    (BoxObject `friction=`), matching the handle bars.
12. **Corner-grasp from yaw jitter**: ±0.6 rad spawn yaw presents the
    59 mm corner diagonal to the pads — grasps came down to luck
    (batches of 3/3 then 0/6 with identical code).  Fix: grasp rotation
    Rz(live yaw)·top-down so the pads always square to the faces, plus
    reduced yaw jitter to ±0.12 rad.  (A 180°-flipped-orientation IK
    fallback was also tried for wrist-limit escapes and made things far
    worse — reverted.)
13. **Wrist yaw oscillation drops the cube**: the transit tracked the
    carried cube's LIVE quat — the pendulum swings the cube, the wrist
    follows, the swing grows until the cube is flung.  Fix: freeze the
    grasp yaw once per phase (captured at rest).
14. **Wrist-branch switch flings the grasp**: letting IK fall back to
    unseeded/home branches mid-carry produced wild joint sweeps.
    Fix: `seeded_only=True` on carry waypoints (2× iterations, no
    fallback).
15. **The systemic fix — verified phases with retries**: raw single-pass
    policy plateaued at ~50–75% with several ~15% flake modes (residual
    grasp miss, bounce-out, jam).  Restructured the expert into
    per-phase builders + geometric verifiers (lid ≥ 100° open, drawer ≥
    8 cm out, cube resting in its zone) with live-state retries (≤ 3).
    Result: **8/8 = 100%** on the M3 acceptance gate.

## Round 5 — M3.5 penetration audit

16. **Audit caught a real retry defect**: on a failed blue place, the
    blind retry descent pressed `gripper0_right_hand_collision` into the
    nudged cube (9.0 mm, 5 steps) — the only arm-link hit ever recorded.
    Fix: `recovery_lift` — climb to carry height above the cube's live
    position and settle 1.5 s before any retry descent.  Audit now passes
    with max penetration 0.6 mm (solver noise) and zero arm-link hits.
17. **Whitelist naming**: robosuite names cube collision geoms
    `RedCube_g0` (not `RedCube_main`/`RedCube_geom`) — the audit's
    whitelist missed the legitimate grasp/table contacts.  Fixed; the
    report is now clean.

## Round 6 — M4 demo capture

18. **agentview camera on the wrong side**: the +y-side camera looked at
    the box BACK — the open lid blocked the whole task in the video.
    Iterated three framings (static M1 render previews between tries) to
    the final front-left shoulder view: box dominant, robot enters from
    frame left, sideview carries the detail.

# Iteration Log — bottle-in-tray scene-to-data pipeline

Every entry below is a failure the pipeline itself caught, fixed **in the
builder/IR** (not in the backend), followed by an end-to-end rerun.

## M0 — visual iteration (renders/m0)

1. **Over-exposure** — first render was blown out (table pure white, red
   sphere read as pink, tray read as pale periwinkle). Fix: sun 4.5→2.2,
   fill 180→50, ceiling 120→30, world strength 0.55→0.22, exposure 0.25→0.
2. **AgX view transform washed task colors pastel** — bad for VLA color
   separation. Fix: force `Standard` view transform; saturated green/blue/red
   now render true to albedo.
3. **Checkerboard contrast too weak** on a light table. Fix: checker colors
   0.80/0.94 → 0.71/0.95 (final pass: 0.60/0.93 after MuJoCo obs also proved
   washed out).

## M1 — physics

4. **Stale hardcoded table height in the physics self-check** — after the
   table-height redesign the check compared the bottle against z=0.82 while
   the bottle correctly settled at 0.69. Fix: the checker now derives
   `table_top_z` / expected heights from the IR, not constants.

## M1.5 — reachability

5. Initial layout passed, but the checker was extended to validate **all 4
   corners of each object's jitter rectangle** (not just the nominal point),
   so every random reset is guaranteed reachable.

## M2 — robosuite task

6. **Gripper opening metric cancelled to zero**: robosuite's Panda finger
   slides are +[0, 0.04] and −[0.04, 0]; summing them always gives ~0, which
   both hid the open/closed state and broke the released-gripper success
   term. Fix: opening = |j1| + |j2| (0.08 open, ≈0.05 closed on the 5 cm
   bottle); "released" = no finger↔bottle contact AND opening > 50 % of the
   open reference captured at reset.
7. **robosuite default reset posture crouches at floor level** (eef at
   z≈0.10, facing +x) — for a floor-mounted arm at a table edge this is a
   terrible IK seed. Fix: IR now carries `base_yaw_deg: 90` (face the task
   area) and a FK-validated collision-free `reset_home_joints`
   `[0, −0.5, 0, −1.4, 0, 2.0, 0.78]` (eef ≈0.39 m in front, z≈0.94, palm
   down), applied in `_reset_internal`; fingers opened at reset.

## M3 — expert (the big one)

8. **Floor-mounted Panda vs a 0.75 m table is geometrically impossible**: the
   Panda's elbow pivot sits at 0.33 m and the elbow can never rise above
   ≈0.65 m, so the forearm crossing the table edge (top 0.75 m) always
   tunnels through it. IK "solved" the grasp with the upper arm inside the
   tabletop; execution pressed the arm against the edge with 69 N·m contact
   torque and the fingers closed 7 cm short of the bottle. The reference
   pipeline had silently dodged this by mounting the expert on the tall
   RethinkMount. **Fix in the IR: lower the tabletop to 0.62 m** (low work
   bench) so the arm hurdles the edge with the elbow above the surface.
9. **Layout too far from a floor base**: even at 0.62 m the original bottle
   position (0.60 m horizontal reach) produced only table-edge-grazing IK
   solutions. **Fix in the IR: bottle (0.02, −0.08), tray (0.20, −0.26)**
   (0.26–0.46 m horizontal reach). After this, a multi-seed scan shows
   **10/10 collision-free IK solutions for every task-critical target**.
10. **mink IK has no collision awareness** — solutions self-collide
    (link5|link7) and graze the table. Fix: solutions are checked in-kinematics
    (`_arm_collisions`) and rejected; retries run from live state, reset home,
    and 6 perturbed posture seeds, biased by a `mink.PostureTask` toward the
    elbow-up family.
11. **EE grasp-band semantics**: robosuite's Panda finger *collision* geoms
    are small tip pads sitting ≈at the EE frame height (pitfall #13: palm and
    fingertips nearly coplanar). EE target = bottle center + 0.04 m puts the
    pads on the bottle's upper section, clear of the table and the tray rim.
12. **First acceptance run 7/8 (88 %)** — the single failure dropped the
    bottle 5 mm outside the strict y-tolerance (zone = tray opening minus the
    bottle radius, i.e. "whole bottle inside the opening"). Fix: gentler
    release — drop clearance 12→6 mm, per-waypoint speed cap 0.08 rad/step for
    the tray descent, settle dwell before opening. Re-run: **10/10 (100 %)**.

## M3.5 — penetration audit

13. First pass after the fixes: **zero arm-link collisions, max penetration
    1.2 mm** (bottle↔tray bottom/wall placement contacts only) — PASS.

## M4 — dataset

14. **Renderer framebuffer limit**: creating the 640×480 review renderer with
    (height, width) swapped raised "Image height 640 > framebuffer height
    480". Fix: honor (height, width) = (480, 640).
15. **Overexposed obs images** (checkerboard invisible, bottle green too
    close to the Panda's robot green). Fix in the IR: table checker 0.60/0.93,
    bottle rgba (0.02, 0.36, 0.10), arena key light diffuse 1.0→0.72. All
    backends recompiled from the IR.
16. **MuJoCo's DEFAULT headlight stacks on the key light** (~0.7 diffuse
    extra), still clipping the light table squares. Fix: pin the headlight
    explicitly in the arena (ambient 0.22, diffuse 0) and key light 0.80,
    light checker square 0.88 — checkerboard clearly visible, no clipping.
17. **Agentview camera placement**: (a) the original front camera put the
    task area behind the robot's own arm; (b) steeper variants still had the
    horizon in frame or the home-pose gripper dominating; (c) the sightline
    from any front camera passes through the work area where the arm
    operates. Fix in the IR: LIBERO-style side view
    `pos=[-0.78,-0.28,1.08], target=[0.12,-0.18,0.64]` — bottle prominent,
    tray visible, arm enters from the right, horizon at frame bottom.

## Final gate evidence

| Gate | Result |
| --- | --- |
| M0 renders + IR | 3 receipts + `spec/scene_spec.json` |
| M1 physics settle | bottle z 0.6896 (expected 0.690), speeds ≈0, no AABB overlap |
| M1.5 reachability | bottle jitter rect 0.39–0.46 m, tray 0.22–0.30 m ⊂ (0.15, 0.78] |
| M2 resets (15) | objects on surfaces, success False at init, jitter bounds held |
| M3 expert | 10/10 = 100 % (gate ≥80 % over ≥6) |
| M3.5 penetration | 0 arm-link collisions, max 1.2 mm |
| M4 dataset | 8 episodes written, read-back PASS, all success=True, MP4 saved |

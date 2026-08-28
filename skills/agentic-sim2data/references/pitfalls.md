# Pitfalls — real bugs from the reference implementation

Every entry below was an actual failure encountered while building the
reference pipeline. Check this list before debugging "impossible" behavior.

## robosuite 1.5.x specific

1. **Silent ±1 action clamping.** The env clips actions to the controller's
   `input_min`/`input_max` (default ±1). Absolute world-space targets (z≈0.9 m)
   and joint angles (±2.9 rad) get silently truncated — the robot "can't
   reach" targets it physically can. Fix: set `input_max`/`input_min` to the
   physical range per controller segment (e.g. ±4.5 rad for 7 arm joints;
   world pos ±10 and rotvec ±4 for absolute OSC).
2. **Gripper action dimension.** With `JOINT_POSITION` the action is
   7 arm joints + 1 gripper = 8 dims; the gripper is index 7. Writing the
   open/close command into index 6 twists the last wrist joint instead. Print
   `env.action_dim` and `env.action_spec` before assuming.
3. **`env.reset()` (hard_reset) rebuilds the entire `MjModel`.** Anything that
   captured the old model — mink `Configuration`, custom renderers, cached
   references — must be re-bound at the start of every episode. Symptom:
   stage works once, then all following episodes misbehave or objects vanish
   from renders. (`obj_body_id` maps must be rebuilt in `_setup_references`.)
4. **robosuite's own offscreen renderer can hold a stale model** after
   hard_reset: observations keep rendering the previous scene (new objects
   missing, old lights). Workaround: render observations with
   `mujoco.Renderer` bound to `env.sim.model._model` directly.
5. **`hard_reset=True` rebuilds the model on every `reset()`**, so anything
   captured at env construction is stale from episode 1 onward.
6. **`_reset_internal` must re-apply placements.** The base class resets
   robots only; call the placement sampler and write `qpos` (free joints) /
   `body_pos` (fixed geoms) yourself, mirroring `PickPlace._reset_internal`.
7. **RethinkMount is the Panda default base** — a tall pedestal that lifts the
   whole arm ~0.5 m, pushing tabletop targets out of reach. Prefer
   `NullMount` (robot stands on the floor) and place the base near the table
   edge, or shrink the task footprint to what the arm reaches at that height.
8. **Primitive objects override rgba with their own material.** If exact
   colors matter, emit them as arena geoms (with a texture for large flat
   surfaces) instead of `MujocoObject`s.
9. **Blender-style RGB/BGR double conversion.** `mujoco.Renderer.render()`
   returns RGB; `imageio.mimsave` expects RGB; but `cv2.imwrite` expects BGR.
   Pick one convention per output path — double conversion makes every color
   swapped (red box looks blue, orange looks blue-purple).
10. **`load_part_controller_config` returns a single-controller dict that the
    composite factory rejects.** Load `load_composite_controller_config(
    controller="BASIC", robot=...)` and surgically replace the arm segment.
11. **Custom arena lights:** robosuite arenas ship their own lights; delete
    them before adding yours or you get multiple conflicting shadows that bury
    task objects in darkness.

## MuJoCo / geometry

12. **Fingertip vs palm offset is orientation-dependent.** The vertical
    distance between the EE frame and the fingertips changes with wrist
    configuration; never treat it as a constant measured in a different pose.
    A palm target at object-center height can leave fingertips hovering above
    the object (grasping air) or pressing into its top face (blocking the
    close).
13. **Top-down grasping of a short object on a table has a hard floor:** palm
    and fingertips are nearly coplanar, so the palm bottoms out on the table
    before fingers can wrap the object's mid-section. Fix: raise the object on
    a small fixture/plinth so the grasp band sits above the table, or switch
    to a lateral grasp.
14. **Gripper aperture is smaller than the spec sheet.** Measure it in-sim
    (FK distance between finger tips at full open). robosuite PandaGripper
    measures ~5.9 cm, not the advertised 8 cm — a 7 cm cube is ungraspable.
    Reduce the object size in the IR.
15. **Settle-phase penetration.** Objects placed exactly touching a surface
    can penetrate deeply on the first integration steps (placement jitter +
    free fall). Either place with a small gap and let them settle, or exclude
    the settle window from penetration audits / make distractor objects
    static.
16. **Penetration audit is not optional.** Logging every contact per step and
    aggregating max depth per geometry pair finds real defects (objects
    knocked across the table, tunneling, arm collisions) that success-rate
    metrics hide. Whitelist legitimate resting/grasp contact pairs.

## Process

17. **Validation gates drive design.** Reachability pre-checks, penetration
    audits, and reset-distribution tests will force layout/size changes.
    Apply fixes in the IR/builder and rerun everything — never patch the
    backend only, or the two representations drift apart.
18. **Keep a debug-render escape hatch.** When a wrapper's render looks wrong,
    render the same camera with `mujoco.Renderer` directly and compare
    side-by-side. This instantly separates "scene problem" from "render
    pipeline problem" (see #4).

## From the multi-task builds (hinged lid, drawer, bottle-tray)

19. **Blind retry descent hits the nudged object.** When a place fails and the
    object got pushed aside, re-running the same descent trajectory presses the
    gripper palm into the object at its new position (9 mm arm-link hit in the
    reference build). Fix: on retry, first climb to carry height above the
    object's *live* position and settle before descending again.
20. **robosuite collision geom names end in `_g0`** (e.g. `RedCube_g0`), not
    `_main` or `_geom`. Penetration-audit whitelists and contact filters must
    match the actual names or legitimate grasp/table contacts get flagged.
21. **Hinged lids fall back under gravity.** A revolute lid joint without
    damping slams shut mid-trajectory, invalidating "open" states. Give the
    joint damping/friction in the MJCF and express the open/closed success
    condition as an angle threshold, never as velocity.
22. **Agentview camera placement is per-task.** A camera on the opposite side
    of the task filmed the box's back — the open lid blocked the whole scene.
    Iterate framing with cheap M1-style renders before recording, and prefer a
    shoulder view where the robot enters from frame left.
23. **Long-horizon episodes need staged success checks.** For
    open→manipulate→close chains, check each sub-goal (lid open, object placed,
    lid closed) separately during data collection; a single end-state check
    makes failures undebuggable.

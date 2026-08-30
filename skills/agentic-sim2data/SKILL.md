---
name: agentic-sim2data
description: Agentic pipeline that turns natural-language scene requests into physically-validated robot simulation scenes and LIBERO-style manipulation datasets. Use whenever the user wants to generate interactive robot simulation scenes (Blender, MuJoCo, robosuite, Isaac), create synthetic demonstration data for VLA/imitation learning, convert scenes between simulators, build tabletop pick-place tasks with a Franka/Panda or similar arm, or asks about sim-to-real data engines, scene validation gates, penetration/reachability checks, or MimicGen-style trajectory generation — even if they only say "make a robot scene" or "generate some training data for my robot".
---

# Agentic Scene-to-Data Pipeline

Drive the full loop autonomously: natural-language scene request → validated
simulation scene → robot task → expert trajectories → LIBERO-style dataset.
Do not stop at "here is the code" — run every stage, run every validation gate,
and let gate failures drive design fixes until all gates pass.

## The non-negotiable architecture rule

Put an **Intermediate Representation (IR)** — a JSON scene spec — between the
scene authoring tool and the simulator. The IR is the single source of truth:
object names, primitive shapes, dims, poses, mass, rgba, semantics
(`grasp_target` / `distractor` / `container` / `fixture` / `obstacle`), task
definition, success condition, init randomization, robot mount pose, cameras.

- Scene authoring (Blender) *dumps* the IR; the simulator backend *compiles*
  it. Never hand-maintain geometry in two places.
- Any validation failure that forces a design change is fixed **in the builder,
  not in the backend**, then the whole pipeline reruns end-to-end.

## Pipeline stages and gates

Run these in order. Each stage has a hard gate; a failed gate blocks the
pipeline and must be diagnosed before moving on. A reference implementation of
every stage exists in this repository's `pipeline/` directory (see
"Reference implementation" below).

| Stage | Gate (must pass before proceeding) |
| --- | --- |
| M0 Scene build (Blender + IR dump) | scene renders, interaction panel works, IR written |
| M1 IR → MJCF compile | physics self-check: objects settle on surfaces, no pairwise penetration, final velocities ≈ 0 |
| M1.5 Reachability pre-check | every task-critical object within ~0.78 m horizontal reach of the arm base (static decorations exempt) |
| M2 robosuite task wrapper | N random resets: objects stay on surfaces, success stays False at init, jitter bounds respected |
| M3 Expert policy acceptance | success rate ≥ 80% over ≥ 6 randomized episodes |
| M3.5 Penetration audit | zero arm-link collisions; all contact penetration ≤ solver noise (a few mm); no sustained deep penetration |
| M4 Dataset collection | HDF5 written, every episode success=True, obs/action shapes verified by reading the file back |

## Stage notes

### M0 Scene generation (Blender)

- Write one builder script that constructs the whole scene from constants and
  **dumps the IR in the same run**. Do not parse `.blend` files.
- Iterate visually: render fixed cameras → inspect → fix → rebuild. Blockout
  early, polish late. Common visual failures: over/under-exposure, camera
  blocked by walls, no-texture planes misread as vertical surfaces (use
  checkerboard textures on horizontal work surfaces), distractor objects
  colored too close to the background.
- For interactive scenes, embed the interaction script as a text datablock and
  document the one-command launch (`blender scene.blend --python script.py`) —
  never assume scripts auto-run on file open.

### M1 IR → MuJoCo

- Emit only primitives (box/cylinder/sphere) for physics; meshes are a later
  enhancement. Primitives make contacts reliable and validation tractable.
- **Widen `input_max`/`input_min` on any controller fed absolute physical
  units.** robosuite defaults to ±1 and silently clips (see pitfalls).
- Gate: after ≥3 s of simulation, all dynamic bodies rest on their support
  surfaces (height within tolerance), no pairwise AABB overlap, final speeds
  ≈ 0. Render one offscreen frame as a visual receipt.

### M1.5 Reachability pre-check

Before any robot work, compute horizontal base→object distance for every
task-critical object and require it inside ~0.15–0.78 m (Panda). This cheap
check catches layouts that would waste hours of grasping debugging. When it
fails: move the base or the objects — fix the IR, rerun.

### M2 robosuite task

- Subclass the robosuite manipulation env; custom `Arena` subclass emits
  static scene geoms directly into `worldbody` (robosuite primitives override
  rgba with their own materials — strip the `material` attribute if you need
  exact colors, and use a texture for the work surface).
- Dynamic objects via `MujocoObject` subclasses + `SequentialCompositeSampler`
  with fixed or jittered placements; write `_reset_internal` to re-apply
  placements after every reset (hard_reset rebuilds the model).
- Success = task geometric condition (object inside container zone, low
  velocity) implemented in `_check_success`, plus a `reward` method (1.0 on
  success) or `env.step` raises.

### M3 Expert trajectories: offline IK + joint-position control

Preferred recipe (deterministic, minimal control-semantics surface):

1. Plan world-space keypoints relative to detected object poses: hover above
   grasp pose → descend → close → lift → above container → lower → release →
   retreat.
2. For each keypoint, run numerical IK (e.g. `mink`) from the **current live
   state**; on local-minimum failure, retry from the reset joint configuration.
3. Execute with the `JOINT_POSITION` controller in absolute mode, stepping
   joint targets toward the IK solution at a capped per-step delta.
4. Acceptance: success rate ≥ 80% across randomized inits. Add a
   penetration audit stage (see M3.5) before declaring victory.

Calibrate two numbers empirically before tuning anything else: (a) the
palm-to-fingertip vertical offset under the actual grasp orientation, and (b)
the steady-state vertical droop of the controlled EE under gravity — aim the
descent target at `object_center + droop` so fingertips land at the intended
grasp band.

Gripper action lives in its own action dimension (e.g. index 7 when the arm
takes 7); writing it into the last arm joint silently twists the wrist instead
of closing the gripper.

### M3.5 Penetration audit

Run the expert episode while logging every MuJoCo contact per step; aggregate
max penetration depth per geometry pair. Fail on: any arm-link (non-finger)
collision; any pair with sustained penetration > 8 mm or a spike > 25 mm.
Verify end-state geometry (object inside container, not intersecting walls).
Keep a whitelist for legitimate resting/grasp contacts. This stage routinely
finds real defects — treat every failure as a design fix, not noise.

### M4 Standard Outputs (see docs/OUTPUT_SPEC.md for full spec)

Every task must produce ALL of the following upon completion:

| Artifact | Path | Gate |
|---|---|---|
| Scene IR | `spec/scene_spec.json` | schema valid |
| Dataset | `data/<task>/demo.hdf5` | states + model_file + env_args present, terminal done=True |
| Demo video | `rollouts/<task>_demo.mp4` | 640×480 @ 20fps |
| Demo GIF | `images/<task>.gif` | 420px wide, auto-loop |
| Validation report | `docs/validation_<task>.json` | all gates pass |

A missing or invalid artifact blocks release. See `docs/OUTPUT_SPEC.md` for exact schema.

### M4 Dataset (LIBERO drop-in schema)

Write HDF5 directly. For the dataset to be **drop-in loadable by the official
LIBERO scripts** (not just "similar"), every episode needs:

- `model_file` attr: the full MJCF string of the compiled env (serialize
  `env.model.get_xml()` after reset — it embeds robots, arena and objects);
- `states` (T, nq+nv): full sim state per step (`env.sim.get_state()` flattened);
  replay = write qpos/qvel into `MjData` + `mj_forward` — must reproduce the
  recorded EE pose to <5 mm;
- `env_args` attr: JSON with `env_name` (registered in
  `robosuite.environments.base.REGISTERED_ENVS`), `env_kwargs` exactly as
  passed to the env constructor (controller config must be the composite dict
  captured *before* reset — the normalized dict robosuite keeps after reset
  contains internal keys and will fail reconstruction);
- `dones[-1] = True` forced (expert policies that break early leave the last
  step's done=False);
- action semantics documented explicitly (absolute joint-position vs OSC
  delta — downstream training heads differ).

Write HDF5 directly:

```text
data/demo_i/
  attrs: num_samples, success, instruction, model_file (full MJCF), env_args (JSON)
  actions   (T, action_dim)
  states    (T, nq + nv)
  obs/agentview_image (T,H,W,3), robot0_eef_pos/quat, robot0_joint_pos, ...
  dones     (T,)  ← last frame MUST be True
```

Render observations with `mujoco.Renderer` bound to the **live** model — not
through any wrapper that may hold a stale model. Read the file back after
writing and verify shapes and a mid-episode frame. Record one MP4 demo for
human review.

### M4.5 Standard Output Specification

Every task must produce ALL of the following artifacts. A missing or invalid
artifact blocks release. Detailed schema: `docs/OUTPUT_SPEC.md`.

| Artifact | Path | Format | Gate |
|---|---|---|---|
| Scene IR | `spec/scene_spec.json` | JSON (versioned schema) | all required fields present |
| Dataset | `data/<task>/demo.hdf5` | HDF5 (LIBERO drop-in) | states + model_file + env_args + terminal dones |
| Demo video | `rollouts/<task>_demo.mp4` | MP4 (H.264) | 640×480 @ 20fps minimum |
| Demo GIF | `images/<task>.gif` | GIF | 420px wide, auto-loop, ≤ 5 MB |
| Validation report | `docs/validation_<task>.json` | JSON | all gates pass, structured output |

HDF5 per-episode required fields:

| Field | Shape / Type | Notes |
|---|---|---|
| `actions` | (T, action_dim) float64 | action_dim depends on controller; document semantics |
| `states` | (T, nq + nv) float64 | full sim state, frame-accurate replay |
| `dones` | (T,) uint8 | last frame MUST be True |
| `obs/agentview_image` | (T, 480, 640, 3) uint8 | minimum resolution; higher OK |
| `obs/robot0_eef_pos` | (T, 3) float64 | |
| `obs/robot0_eef_quat` | (T, 4) float64 | (w,x,y,z) |
| `obs/robot0_joint_pos` | (T, 7) float64 | |
| `obs/robot0_gripper_qpos` | (T, 2) float64 | |
| attrs `model_file` | MJCF string | full env, compiles via `mujoco.MjModel.from_xml_string` |
| attrs `env_args` | JSON dict | must include `env_name`, `env_kwargs.robots`, `env_kwargs.controller_configs`, `env_kwargs.base_pos` |

Action semantics MUST be declared: `absolute_joint_position` (8-dim) or
`osc_delta` (7-dim). Never mix the two in one dataset.

Dataset size classes:

| Class | Criteria | Example |
|---|---|---|
| validated | ≥ 3 episodes, all success, states + model_file + env_args present, schema verified | `put_red_in_box` (6 eps) |
| preview | Gates pass but < 3 episodes OR states missing | `lid_open` (1 ep) |
| wip | Any gate fails or code incomplete | — |

### M4.5 LIBERO Drop-in Checklist

For the dataset to be loadable by official LIBERO scripts (not just "similar"):

1. `model_file` attr: full MJCF string from `env.model.get_xml()` after reset;
2. `states` (T, nq+nv): full sim state per step, replay = write qpos/qvel into
   `MjData` + `mj_forward`, must reproduce recorded EE pose to < 1 mm;
3. `env_args` attr: JSON with `env_name` (registered in REGISTERED_ENVS),
   `env_kwargs` exactly as passed to the constructor (composite config captured
   *before* reset — the post-reset normalized dict has internal keys);
4. `dones[-1] = True` forced;
5. action semantics documented: absolute joint-position vs OSC delta.

## Reference implementation

The checked-in `pipeline/` directory of this repository is the working
reference (copying the skill alone does not carry the pipeline — clone the
repo)
(`compile_mjcf.py`, `task_put_red_in_box.py`, `expert_ik.py`,
`collect_demos.py`, `test_penetration.py`, and `run_pipeline.py`). Reuse its
structure and constants as a starting point; adapt names, layout, and robot to
the new task.

## Pitfalls

Read `references/pitfalls.md` before writing any stage code — every entry there
was a real bug that cost a debugging cycle.

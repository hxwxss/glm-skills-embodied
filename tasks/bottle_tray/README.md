# bottle_tray_trial — "put the green bottle in the tray"

Agentic scene-to-data pipeline (skill: `agentic-sim2data`): a Panda arm puts
the green bottle on a low checkerboard table into a shallow blue tray.
Blender builds the scene and dumps the JSON IR in the same run; the IR is the
single source of truth for the MuJoCo backend, the robosuite task, the expert
and the dataset.

## One-command reproduction

```bash
cd mujoco_env
python run_pipeline.py          # all 9 stages, exit 0 = every gate passed
```

Stages: Blender build + IR dump (+3 render receipts) → MJCF compile → physics
settle check → reachability pre-check → robosuite reset test → IK expert
acceptance (≥80 %) → penetration audit → HDF5 collection (+MP4) → dataset
read-back verification.

## Scene (from `spec/scene_spec.json`)

| element | spec |
| --- | --- |
| table | 1.9 × 0.95 m light checkerboard, top at **0.62 m** (low bench: a floor-mounted Panda's elbow tops out at ~0.65 m, so 0.75 m tabletops are unreachable without a pedestal — see iteration_log #8) |
| GreenBottle | cylinder r=0.025 h=0.14, 0.15 kg (diameter 0.05 < in-sim gripper aperture ~0.059; tall enough that a top-down grasp needs **no plinth**) |
| Tray | shallow container 0.17 × 0.13 × 0.045, wall 0.012, opening upward, deep blue, jitters ±0.03 m per reset (`body_pos` rewrite) |
| RedSphere | one static red distractor (static: dynamic distractors tunneled during settle in the reference pipeline) |
| Panda | NullMount on the **floor at the table edge** (0.10, −0.50), yaw 90°, IR-defined ready home posture |
| success | whole bottle inside the tray opening (zone = opening − bottle radius), standing band, speed < 0.05 m/s, fingers off the bottle |

## Key findings (full list in `iteration_log.md`)

1. **Floor-mounted Panda vs 0.75 m table is impossible geometry** — the elbow
   can never rise above the tabletop, so the forearm tunnels through the table
   edge; the reference pipeline had silently dodged this with the tall
   RethinkMount. The IR lowers the table to 0.62 m.
2. IK is made collision-aware: solutions are kinematically checked and
   retried (live → home → perturbed seeds) with a PostureTask bias.
3. robosuite finger slides are ±[0, 0.04]: summing them cancels to zero.
4. MuJoCo's default headlight stacks on arena lights and clips light surfaces.

## Layout

```
bottle_tray_scene.blend        Blender scene (visual backend)
build_scene_bottle.py          builder + IR dump (+ M0 renders)
spec/scene_spec.json           ★ IR — single source of truth
mujoco_env/
  compile_mjcf.py              IR → generated/bottle_tray_scene.xml
  test_mjcf_physics.py         M1 settle/penetration/velocity gate
  check_reachability.py        M1.5 Panda workspace gate (jitter corners)
  task_bottle_in_tray.py       M2 robosuite task (+ --reset-test)
  expert_ik.py                 M3 mink IK + JOINT_POSITION expert (+ calib/test/demo)
  test_penetration.py          M3.5 contact-depth audit
  collect_demos.py             M4 HDF5 + MP4
  verify_dataset.py            M4 read-back shape verification
  run_pipeline.py              ★ end-to-end
demos/bottle_tray_demo.hdf5    8 episodes, 100 % success
rollouts/bottle_in_tray_demo.mp4
renders/                       M0 receipts, settle receipt, dataset frames
```

# Validation Gates — Defects Found Autonomously

Every defect below was found by an automated validation gate in the pipeline,
then fixed in the scene IR, then the whole pipeline reran. No human fixes.

| # | Found by | Defect | Fix |
|---|---|---|---|
| 1 | Gripper-aperture check | 7 cm cube > 5.9 cm measured gripper aperture — physically ungraspable | cube resized to 5 cm |
| 2 | Reachability pre-check | all objects 1.57 m from base (arm reach 0.855 m) | task zone re-laid out |
| 3 | Action-range inspection | env silently clipped actions to ±1 — joint angles & world targets truncated | controller ranges opened to physical units |
| 4 | Gripper telemetry | fingers never closed: gripper command written to wrong action dim (6 instead of 7) | index fixed |
| 5 | Top-down camera render | transparent box invisible against checkerboard (color collision) | orange translucent walls + bold rim outline |
| 6 | Penetration audit | distractor ball smashed 19.6 mm through the floor during settle | distractor made static |
| 7 | Camera composition | featureless white table read as a wall — scene looked upside down | near-vertical top-down camera + checkerboard table texture |
| 8 | Tracking telemetry | IK solution's joint configuration diverged from physical joints (7-DOF redundancy) | arrival check switched to EE-space distance |
| 9 | Controller audit | kp=30 too soft for gravity compensation (0.3 m steady-state droop) | kp tuned to 30→300 sweep, settled at 30 |
| 10 | Stale model | hard_reset rebuilds MjModel; mink Configuration and robosuite offscreen renderer both bound to the old instance | rebind at every reset; mujoco.Renderer for all rendering |
| 11 | Env kwargs drift | base_pos, ignore_done, controller — internal keys leaked into env_args and broke robosuite.make reconstruction | canonical pre-reset composite config stored |
| 12 | Joint-name API | raw MjModel lacks mj_id2name; wrapper MjModel expects name strings, not indices | set_joint_qpos("RedCube_joint0", ...) |
| 13 | RGB/BGR double conversion | mujoco.Renderer returns RGB; imageio expects RGB; but cv2 intermediate converted to BGR then re-encoded | removed intermediate conversion; single RGB→file path |

These gates are not decorative: each one caught a real defect that would have
silently produced broken data or a physically impossible scene.

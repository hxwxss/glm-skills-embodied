{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "Embodied Task Output Spec v0.3",
  "description": "Every task must produce ALL of the following artifacts upon completion. A missing or invalid artifact blocks release.",

  "required_outputs": {
    "scene_ir": {
      "path": "spec/scene_spec.json",
      "format": "JSON (scene-spec schema)",
      "required_fields": ["schema", "scene_name", "objects", "robots", "cameras", "task", "provenance"],
      "task_required_fields": ["name", "instruction", "success_condition", "init_randomization"],
      "notes": "Single source of truth. All backends consume this."
    },
    "dataset": {
      "path": "data/<task_name>/demo.hdf5",
      "format": "HDF5 (LIBERO/robosuite compatible)",
      "required_groups": ["data"],
      "required_per_episode": ["actions", "dones", "obs"],
      "required_obs_keys": ["agentview_image", "robot0_eef_pos", "robot0_eef_quat", "robot0_joint_pos", "robot0_gripper_qpos"],
      "optional_obs_keys": ["agentview_depth", "wrist_image", "lid_qpos", "drawer_qpos", "red_cube_pos", "blue_cube_pos"],
      "required_attrs_per_episode": ["num_samples", "success", "instruction", "model_file", "env_args"],
      "required_top_attrs": ["spec_snapshot", "success_rate", "total_episodes"],
      "notes": "model_file = full MJCF string for environment reconstruction. env_args must be sufficient for robosuite.make() to recreate the env."
    },
    "demo_video": {
      "path": "rollouts/<task_name>_demo.mp4",
      "format": "MP4 (H.264), 640x480 @ 20fps minimum",
      "notes": "Single continuous episode showing the full task. No cuts."
    },
    "demo_gif": {
      "path": "images/<task_name>.gif",
      "format": "GIF, 420px wide, auto-loop",
      "notes": "For README embedding. Derived from demo MP4."
    },
    "validation_report": {
      "path": "docs/validation_<task_name>.json",
      "format": "JSON",
      "required_fields": {
        "gates": [
          {"name": "physics_settle", "pass": "bool", "metrics": "dict"},
          {"name": "reachability", "pass": "bool", "objects_checked": "list"},
          {"name": "reset_distribution", "pass": "bool", "episodes_tested": "int"},
          {"name": "penetration_audit", "pass": "bool", "max_depth_mm": "float", "arm_link_collisions": "int"},
          {"name": "expert_success_rate", "pass": "bool", "rate": "float", "episodes": "int"},
          {"name": "states_replay", "pass": "bool", "worst_ee_error_mm": "float"},
          {"name": "hdf5_schema", "pass": "bool", "episodes_verified": "int"}
        ],
        "all_pass": "bool",
        "timestamp": "ISO 8601",
        "git_sha": "string"
      }
    }
  },

  "standard_episode_schema": {
    "actions": "(T, action_dim) float64 — action_dim depends on controller",
    "dones": "(T,) uint8 — last frame must be True",
    "obs/agentview_image": "(T, 480, 640, 3) uint8",
    "obs/robot0_eef_pos": "(T, 3) float64",
    "obs/robot0_eef_quat": "(T, 4) float64 (w,x,y,z)",
    "obs/robot0_joint_pos": "(T, 7) float64",
    "obs/robot0_gripper_qpos": "(T, 2) float64",
    "states": "(T, nq + nv) float64 — full sim state for frame-accurate replay"
  },

  "quality_thresholds": {
    "expert_success_rate": ">= 0.80 across randomized inits",
    "max_penetration_depth_mm": "<= 4.1 (solver noise)",
    "arm_link_collisions": 0,
    "states_replay_ee_error_mm": "<= 1.0",
    "reset_distribution_jitter": "within ±2× specified jitter",
    "initial_success_rate": "0.0 (task must not be pre-solved)"
  },

  "task_status_levels": {
    "validated": "All gates pass, >= 3 episodes, states present, schema verified",
    "preview": "Gates pass but < 3 episodes OR states missing",
    "wip": "Any gate fails or code incomplete"
  }
}

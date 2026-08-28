# lid_open — "flip open the lid of the storage box and leave it open"

Articulated preview task: a storage box whose lid is a real revolute joint
(damping-tuned so it stays open). The Panda hooks the handle bar, flips the
lid past its limit, and holds it open.

- Success: lid angle at its joint limit, angular velocity ≈ 0, held.
- Dataset: [`../../data/lid_open/demo.hdf5`](../../data/lid_open/demo.hdf5) — 1 episode *(preview; dataset
  scaling in progress)*
- Video: [`rollouts/lid_open_demo.mp4`](rollouts/lid_open_demo.mp4)
- Pipeline: `mujoco_env/run_pipeline.py` (7 validation stages, same gates as
  the reference task)

Built with the [`agentic-sim2data`](../../skills/agentic-sim2data/SKILL.md)
skill. Pitfall that shaped this build: an undamped revolute lid falls back
under gravity — the joint carries damping and the success check is an angle
threshold, never velocity alone.

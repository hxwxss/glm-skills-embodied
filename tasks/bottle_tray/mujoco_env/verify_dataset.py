# -*- coding: utf-8 -*-
"""
verify_dataset.py — M4 read-back verification of the HDF5 dataset
==================================================================

Reopens the written file and asserts, for every episode:
  * success=True in attrs
  * actions            (T,8)
  * obs/agentview_image (T,256,256,3), uint8 range
  * obs/robot0_eef_pos (T,3), robot0_eef_quat (T,4),
    robot0_joint_pos (T,7), robot0_gripper_qpos (T,2)
  * dones              (T,)
Extracts one mid-episode frame as a PNG receipt.

Usage: python verify_dataset.py [--hdf5 ../demos/bottle_tray_demo.hdf5]
Exit 0 = PASS.
"""

import argparse
import os
import sys

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

EXPECTED = {
    "actions": ("T", 8),
    "obs/agentview_image": ("T", 256, 256, 3),
    "obs/robot0_eef_pos": ("T", 3),
    "obs/robot0_eef_quat": ("T", 4),
    "obs/robot0_joint_pos": ("T", 7),
    "obs/robot0_gripper_qpos": ("T", 2),
    "dones": ("T",),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdf5", default=os.path.join(
        HERE, "..", "demos", "bottle_tray_demo.hdf5"))
    ap.add_argument("--frame-out", default=os.path.join(
        HERE, "..", "renders", "dataset_midframe.png"))
    args = ap.parse_args()

    failures = []
    print("=" * 62)
    print("DATASET READ-BACK:", args.hdf5)
    print("=" * 62)
    with h5py.File(args.hdf5, "r") as f:
        n_ep = len(f["data"])
        print(f"[ok] file opens; episodes={n_ep}; "
              f"instruction='{f.attrs['instruction']}'")
        if n_ep < 6:
            failures.append(f"only {n_ep} episodes (<6)")
        for name in f["data"]:
            g = f["data"][name]
            T = int(g.attrs["num_samples"])
            if not bool(g.attrs["success"]):
                failures.append(f"{name}: success=False")
            for key, shape_spec in EXPECTED.items():
                if key not in g:
                    failures.append(f"{name}: missing {key}")
                    continue
                ds = g[key]
                want = tuple(T if s == "T" else s for s in shape_spec)
                if ds.shape != want:
                    failures.append(f"{name}: {key} shape {ds.shape} != {want}")
            # image sanity: non-degenerate uint8 frames
            img = g["obs/agentview_image"]
            if img.dtype != np.uint8:
                failures.append(f"{name}: agentview_image dtype {img.dtype}")
            mid = img[T // 2]
            if mid.std() < 5:
                failures.append(f"{name}: mid-episode frame is flat/degenerate")
            # quaternion norm sanity
            q = g["obs/robot0_eef_quat"][()]
            norms = np.linalg.norm(q, axis=1)
            if not np.allclose(norms, 1.0, atol=0.02):
                failures.append(f"{name}: eef quat norms off "
                                f"[{norms.min():.3f},{norms.max():.3f}]")
            print(f"[ok] {name}: T={T} success={bool(g.attrs['success'])} "
                  f"shapes verified; mid-frame std={mid.std():.1f}")
            if name == sorted(f["data"].keys())[len(list(f['data'])) // 2]:
                try:
                    import cv2
                    cv2.imwrite(args.frame_out,
                                cv2.cvtColor(mid, cv2.COLOR_RGB2BGR))
                    print(f"[ok] mid-episode frame receipt -> {args.frame_out}")
                except Exception as e:
                    print(f"[warn] frame receipt failed: {e}")
        sr = float(f.attrs["success_rate"])
        print(f"[{'ok' if sr >= 0.999 else 'FAIL'}] dataset success_rate="
              f"{sr:.2f} (all episodes must be success=True)")

    print("-" * 62)
    if failures:
        print("DATASET_VERIFY: FAILED")
        for x in failures:
            print("  -", x)
        sys.exit(1)
    print("DATASET_VERIFY: PASS")
    sys.exit(0)


if __name__ == "__main__":
    main()

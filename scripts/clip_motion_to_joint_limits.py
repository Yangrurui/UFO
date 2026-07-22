#!/usr/bin/env python3
"""Clamp Adam SP retargeted motion data to the robot's real joint limits.

Background
----------
The retargeted motion (`*.pkl`) was produced with joint ranges that exceed
Adam SP's mechanical limits (very likely G1's ranges were reused). During
BFM-Zero training the reference is derived entirely from ``pose_aa``:

    pose_aa[:, 1:] == dof[..., None] * joint_axis      (verified, exact)
    dof_pos (FK)   == proj(pose_aa[:, 1:], joint_axis)

The pkl also stores a redundant ``dof`` field (unused by the loader, since it
looks for key ``dof_pos``), which we keep consistent anyway.

This script clamps every actuated joint angle to ``[lo, hi]`` taken from the
compiled MuJoCo model (``jnt_range``) and rebuilds ``pose_aa`` accordingly.
Root orientation (``pose_aa[:, 0]``), ``root_trans_offset``, ``root_rot`` and
all other fields are left untouched.

Usage
-----
    python scripts/clip_motion_to_joint_limits.py \
        --pkl humanoidverse/data/adam_sp_lafan_29dof_10s-clipped.pkl \
        --xml humanoidverse/data/robots/adam_sp_mjlab/adam_sp.xml \
        [--margin-deg 0.0] [--dry-run] [--out PATH]

By default it writes back in place after saving a ``*.pre_limitfix.pkl`` backup.
"""
import argparse
import os
import numpy as np


def get_joint_axes_and_ranges(xml_path):
    import mujoco
    m = mujoco.MjModel.from_xml_path(xml_path)
    names, axes, ranges = [], [], []
    for i in range(m.njnt):
        if m.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        names.append(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i))
        axes.append(m.jnt_axis[i].copy())
        ranges.append(m.jnt_range[i].copy())
    return names, np.asarray(axes, dtype=np.float64), np.asarray(ranges, dtype=np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", default="humanoidverse/data/adam_sp_lafan_29dof_10s-clipped.pkl")
    ap.add_argument("--xml", default="humanoidverse/data/robots/adam_sp_mjlab/adam_sp.xml")
    ap.add_argument("--out", default=None, help="output path (default: overwrite --pkl in place)")
    ap.add_argument("--margin-deg", type=float, default=0.0,
                    help="shrink each limit inward by this many degrees to avoid boundary saturation")
    ap.add_argument("--no-backup", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="only report stats, do not write")
    args = ap.parse_args()

    import joblib

    names, axes, ranges = get_joint_axes_and_ranges(args.xml)
    ndof = len(names)
    margin = np.deg2rad(args.margin_deg)
    lo = ranges[:, 0] + margin
    hi = ranges[:, 1] - margin
    assert np.all(lo <= hi), "margin too large: crossed a joint range"

    print(f"[info] {ndof} actuated joints; margin={args.margin_deg} deg")
    print(f"[info] loading {args.pkl} ...")
    data = joblib.load(args.pkl)
    keys = list(data.keys())
    print(f"[info] {len(keys)} clips")

    # sanity: check pose_aa/dof/axis relation on first clip
    c0 = data[keys[0]]
    pj0 = np.asarray(c0["pose_aa"])[:, 1:1 + ndof, :]
    proj0 = np.einsum("tjc,jc->tj", pj0, axes)
    err = np.abs(np.asarray(c0["dof"]) - proj0).max()
    recon_err = np.abs(pj0 - np.asarray(c0["dof"])[..., None] * axes[None]).max()
    print(f"[check] |dof - proj(pose_aa,axis)|={err:.2e}  |pose_aa - dof*axis|={recon_err:.2e}")
    assert err < 1e-4 and recon_err < 1e-4, "pose_aa/dof/axis relation broken; aborting"

    per_joint_frames = np.zeros(ndof, dtype=np.int64)
    per_joint_max_over = np.zeros(ndof)  # max radians outside limit (pre-clip)
    total_frames = 0

    for k in keys:
        c = data[k]
        dof = np.asarray(c["dof"], dtype=np.float64)          # (T, ndof)
        pose_aa = np.asarray(c["pose_aa"], dtype=np.float64)  # (T, 30, 3)
        T = dof.shape[0]
        total_frames += T

        over_lo = np.maximum(lo[None] - dof, 0.0)
        over_hi = np.maximum(dof - hi[None], 0.0)
        over = np.maximum(over_lo, over_hi)                    # (T, ndof)
        per_joint_frames += (over > 1e-6).sum(axis=0)
        per_joint_max_over = np.maximum(per_joint_max_over, over.max(axis=0))

        dof_c = np.clip(dof, lo[None], hi[None])
        pose_aa[:, 1:1 + ndof, :] = dof_c[..., None] * axes[None]

        c["dof"] = dof_c.astype(np.float32)
        c["pose_aa"] = pose_aa.astype(np.float32)

    print(f"\n[stats] total frames across dataset: {total_frames}")
    print(f"{'joint':30s} {'lo(deg)':>8s} {'hi(deg)':>8s} {'#frames_clipped':>16s} {'maxover(deg)':>13s}")
    order = np.argsort(-per_joint_frames)
    for j in order:
        if per_joint_frames[j] == 0:
            continue
        print(f"{names[j]:30s} {np.degrees(ranges[j,0]):8.1f} {np.degrees(ranges[j,1]):8.1f} "
              f"{per_joint_frames[j]:16d} {np.degrees(per_joint_max_over[j]):13.2f}")
    n_clipped = int((per_joint_frames > 0).sum())
    print(f"\n[stats] {n_clipped}/{ndof} joints had at least one out-of-limit frame")

    if args.dry_run:
        print("[dry-run] no file written")
        return

    out = args.out or args.pkl
    if out == args.pkl and not args.no_backup:
        bak = args.pkl.rsplit(".pkl", 1)[0] + ".pre_limitfix.pkl"
        if not os.path.exists(bak):
            print(f"[info] backing up original -> {bak}")
            os.replace(args.pkl, bak) if False else __import__("shutil").copyfile(args.pkl, bak)
        else:
            print(f"[info] backup already exists, keeping it: {bak}")
    print(f"[info] writing clamped data -> {out}")
    joblib.dump(data, out)
    print("[done]")


if __name__ == "__main__":
    main()

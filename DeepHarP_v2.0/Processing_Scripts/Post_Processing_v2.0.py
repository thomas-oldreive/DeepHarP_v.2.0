#!/usr/bin/env python3
"""
DeepHarP v2.0 - post-processing of hippocampal segmentations.

For each subject with both a left and a right DeepHarP segmentation, this script:
  1. Cleans each mask by keeping only its largest connected component,
     which removes small disconnected bits of noise.
  2. Puts the cleaned masks back into the subject's registered (_Aladin)
     image space, undoing the crop and flip done by main.sh.
  3. Combines them into one label image (1 = left, 2 = right).
  4. Writes left, right and total hippocampal volumes (cc) to a CSV file.

Requirements:
  - NiftyReg (reg_resample) on your PATH
  - The _Aladin.nii and _regf3d_cpp.nii files that main.sh writes to the
    Brains folder (they must not be deleted before running this script)

Usage:
  python3 Post_Processing_v.2.0.py --predictions <predictions folder> --brains <Brains folder> --out <output folder>
"""

import os
import re
import csv
import shutil
import argparse
import tempfile
import subprocess
import numpy as np
import nibabel as nib
from scipy.ndimage import label

# Matches DeepHarP output names such as  <subject>_ROIFineLeft_Segm.nii.gz
SEGM_PATTERN = re.compile(r"^(.+)_ROIFine(Right|Left)_Segm\.nii(\.gz)?$")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Clean DeepHarP hippocampal masks, combine left and right, "
                    "and compute hippocampal volumes."
    )
    parser.add_argument("--predictions", required=True,
                        help="Folder containing the DeepHarP segmentations "
                             "(searched recursively)")
    parser.add_argument("--brains", required=True,
                        help="DeepHarP Brains folder with the _Aladin and "
                             "_regf3d_cpp files from main.sh")
    parser.add_argument("--atlases", default=None,
                        help="DeepHarP Atlases folder "
                             "(default: the Atlases folder next to --brains)")
    parser.add_argument("--out", required=True,
                        help="Output folder for cleaned masks, combined masks "
                             "and the volumes CSV")
    return parser.parse_args()

def cleanup_mask(data):
    """Keep only the largest connected component of a binary mask."""
    labeled_mask, num_features = label(data > 0)
    if num_features <= 1:
        return (data > 0).astype(np.uint8)
    sizes = np.bincount(labeled_mask.ravel())[1:]
    largest_label = np.argmax(sizes) + 1
    return (labeled_mask == largest_label).astype(np.uint8)

def bbox_from_binary(mask_xyz):
    idx = np.argwhere(mask_xyz > 0)
    if idx.size == 0:
        return None
    mins = idx.min(axis=0)
    maxs = idx.max(axis=0)
    return tuple(mins.tolist() + maxs.tolist())

def apply_flips(arr_xyz, flip_axes):
    out = arr_xyz
    if 0 in flip_axes:
        out = out[::-1, :, :]
    if 1 in flip_axes:
        out = out[:, ::-1, :]
    if 2 in flip_axes:
        out = out[:, :, ::-1]
    return out

def choose_best_flip(seg_xyz, cropped_dil_xyz):
    """Find the flip that best aligns the segmentation with the dilated atlas region."""
    seg = seg_xyz > 0
    ref = cropped_dil_xyz > 0

    best_axes = tuple()
    best_overlap = -1

    candidates = [
        tuple(),
        (0,), (1,), (2,),
        (0, 1), (0, 2), (1, 2),
        (0, 1, 2),
    ]

    for axes in candidates:
        cand = apply_flips(seg, axes)
        overlap = int(np.sum(cand & ref))
        if overlap > best_overlap:
            best_overlap = overlap
            best_axes = axes

    return best_axes, best_overlap

def run_reg_resample(ref_img, flo_img, trans_cpp, out_img):
    cmd = [
        "reg_resample",
        "-ref", ref_img,
        "-flo", flo_img,
        "-trans", trans_cpp,
        "-res", out_img,
        "-inter", "0",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(
            f"reg_resample failed:\n"
            f"CMD: {' '.join(cmd)}\n"
            f"STDERR:\n{r.stderr}\n"
            f"STDOUT:\n{r.stdout}"
        )

def save_uint8_nifti(data, ref_img, out_path):
    img = nib.Nifti1Image(data.astype(np.uint8), ref_img.affine, ref_img.header)
    img.set_data_dtype(np.uint8)
    nib.save(img, out_path)

def find_existing(path_without_ext):
    """Return path.nii or path.nii.gz, whichever exists, else None."""
    for ext in (".nii", ".nii.gz"):
        if os.path.exists(path_without_ext + ext):
            return path_without_ext + ext
    return None

def find_subject_masks(predictions_dir):
    subjects = {}
    for root, _, files in os.walk(predictions_dir):
        for fname in files:
            match = SEGM_PATTERN.match(fname)
            if not match:
                continue
            subject_id = match.group(1)
            side = match.group(2).lower()
            subjects.setdefault(subject_id, {"left": None, "right": None})
            subjects[subject_id][side] = os.path.join(root, fname)
    return subjects

def main():
    args = parse_args()

    brains_dir = os.path.abspath(args.brains)
    atlases_dir = os.path.abspath(
        args.atlases or os.path.join(brains_dir, os.pardir, "Atlases")
    )
    out_dir = os.path.abspath(args.out)

    right_atlas_dil = os.path.join(atlases_dir, "RightHippocampusBinDil.nii.gz")
    left_atlas_dil = os.path.join(atlases_dir, "LeftHippocampusBinDil.nii.gz")

    clean_roi_dir = os.path.join(out_dir, "roi_clean")
    combined_dir = os.path.join(out_dir, "combined_masks")
    csv_path = os.path.join(out_dir, "hippocampal_volumes.csv")

    for p in [right_atlas_dil, left_atlas_dil]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing atlas file: {p}")

    if shutil.which("reg_resample") is None:
        raise RuntimeError("reg_resample (NiftyReg) not found in PATH")

    subjects = find_subject_masks(args.predictions)
    if not subjects:
        print(f"No matching masks found in: {args.predictions}", flush=True)
        return

    os.makedirs(clean_roi_dir, exist_ok=True)
    os.makedirs(combined_dir, exist_ok=True)

    volumes = {}

    for subject_id in sorted(subjects):
        print(f"\n=== {subject_id} ===", flush=True)

        left_path = subjects[subject_id]["left"]
        right_path = subjects[subject_id]["right"]

        if left_path is None or right_path is None:
            missing_side = "left" if left_path is None else "right"
            print(f"Skipping: missing {missing_side} segmentation", flush=True)
            continue

        aladin = find_existing(os.path.join(brains_dir, f"{subject_id}_Aladin"))
        cpp = find_existing(os.path.join(brains_dir, f"{subject_id}_regf3d_cpp"))

        if aladin is None or cpp is None:
            print("Skipping: missing registration files in the Brains folder:", flush=True)
            if aladin is None:
                print(f"  {subject_id}_Aladin.nii(.gz)", flush=True)
            if cpp is None:
                print(f"  {subject_id}_regf3d_cpp.nii(.gz)", flush=True)
            continue

        # 1. Clean each mask: keep the largest connected component
        left_img = nib.load(left_path)
        right_img = nib.load(right_path)

        left_clean = cleanup_mask(left_img.get_fdata() > 0)
        right_clean = cleanup_mask(right_img.get_fdata() > 0)

        save_uint8_nifti(
            left_clean, left_img,
            os.path.join(clean_roi_dir, f"{subject_id}_ROIFineLeft_Segm_clean.nii.gz"),
        )
        save_uint8_nifti(
            right_clean, right_img,
            os.path.join(clean_roi_dir, f"{subject_id}_ROIFineRight_Segm_clean.nii.gz"),
        )

        # 2. Put the cleaned masks back into the registered (_Aladin) space
        aladin_img = nib.load(aladin)
        full_shape = aladin_img.shape[:3]
        full_L = np.zeros(full_shape, dtype=np.uint8)
        full_R = np.zeros(full_shape, dtype=np.uint8)

        tmp_dir = tempfile.mkdtemp(prefix=f"deepharp_uncrop_{subject_id}_")
        try:
            tmp_right_dil = os.path.join(tmp_dir, "right_dil.nii.gz")
            tmp_left_dil = os.path.join(tmp_dir, "left_dil.nii.gz")

            run_reg_resample(aladin, right_atlas_dil, cpp, tmp_right_dil)
            run_reg_resample(aladin, left_atlas_dil, cpp, tmp_left_dil)

            right_dil = (nib.load(tmp_right_dil).get_fdata() > 0).astype(np.uint8)
            left_dil = (nib.load(tmp_left_dil).get_fdata() > 0).astype(np.uint8)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        rb = bbox_from_binary(right_dil)
        lb = bbox_from_binary(left_dil)
        if rb is None or lb is None:
            print("Skipping: dilated atlas region is empty, cannot locate the crop.", flush=True)
            continue

        rx1, ry1, rz1, rx2, ry2, rz2 = rb
        lx1, ly1, lz1, lx2, ly2, lz2 = lb

        cropped_right_dil = right_dil[rx1:rx2 + 1, ry1:ry2 + 1, rz1:rz2 + 1]
        cropped_left_dil = left_dil[lx1:lx2 + 1, ly1:ly2 + 1, lz1:lz2 + 1]

        if right_clean.shape != cropped_right_dil.shape:
            print(f"Skipping: right shape mismatch, mask {right_clean.shape} "
                  f"vs crop {cropped_right_dil.shape}", flush=True)
            continue
        if left_clean.shape != cropped_left_dil.shape:
            print(f"Skipping: left shape mismatch, mask {left_clean.shape} "
                  f"vs crop {cropped_left_dil.shape}", flush=True)
            continue

        best_axes_R, _ = choose_best_flip(right_clean, cropped_right_dil)
        best_axes_L, _ = choose_best_flip(left_clean, cropped_left_dil)

        full_R[rx1:rx2 + 1, ry1:ry2 + 1, rz1:rz2 + 1] = apply_flips(right_clean, best_axes_R)
        full_L[lx1:lx2 + 1, ly1:ly2 + 1, lz1:lz2 + 1] = apply_flips(left_clean, best_axes_L)

        # 3. Combine into one label image: 1 = left, 2 = right
        combined = np.zeros(full_shape, dtype=np.uint8)
        combined[full_L > 0] = 1
        combined[full_R > 0] = 2

        save_uint8_nifti(
            combined, aladin_img,
            os.path.join(combined_dir, f"{subject_id}_hippo_mask_aladin.nii.gz"),
        )

        # 4. Volumes in cc
        voxvol = float(np.prod(aladin_img.header.get_zooms()[:3]))
        left_vol = int(np.sum(combined == 1)) * voxvol / 1000.0
        right_vol = int(np.sum(combined == 2)) * voxvol / 1000.0
        total_vol = left_vol + right_vol

        volumes[subject_id] = (right_vol, left_vol, total_vol)

        print(f"L={left_vol:.3f} cc, R={right_vol:.3f} cc, Total={total_vol:.3f} cc",
              flush=True)

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "subject_id",
            "right_hippocampus_volume_cc",
            "left_hippocampus_volume_cc",
            "total_hippocampal_volume_cc",
        ])
        for sid in sorted(volumes):
            right_vol, left_vol, total_vol = volumes[sid]
            writer.writerow([sid, f"{right_vol:.3f}", f"{left_vol:.3f}", f"{total_vol:.3f}"])

    print(f"\nProcessed {len(volumes)} of {len(subjects)} subjects", flush=True)
    print(f"Volumes CSV     : {csv_path}", flush=True)
    print(f"Clean ROI masks : {clean_roi_dir}", flush=True)
    print(f"Combined masks  : {combined_dir}", flush=True)

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
DeepHarP v2.1 - pre-processing of T1-weighted MRI scans.

For each scan, this script:
  1. Sets low-intensity background voxels to zero (see --threshold, adjust as needed).
  2. Corrects intensity inhomogeneity with N4 bias field correction (SimpleITK).
  3. Removes the skull with FSL BET.

It then measures the brain volume of every skull-stripped scan in the output
folder and writes them to brain_volumes.csv in that folder.

Outputs (in the --out folder):
  <scan name>_processed_Bet.nii.gz              skull-stripped scan, ready for the DeepHarP Brains folder
  masks/<scan name>_processed_Bet_mask.nii.gz   binary BET brain mask, used for the volume
  brain_volumes.csv                             Patient ID, Timepoint, Brain Volume (cc)

Patient ID and timepoint are read from the scan file name. By default names
follow PREVENT_<3-digit patient ID>_<timepoint>, with the timepoint BL or N_yr:
  PREVENT_001_BL.nii    -> patient 001, timepoint BL
  PREVENT_001_1_yr.nii  -> patient 001, timepoint 1_yr
Use --name-pattern to match a different naming scheme, for example
  --name-pattern '^(?P<patient>[^_]+)_(?P<timepoint>[^_]+)'  for patient_timepoint_...

Requirements:
  - FSL on your PATH: bet, plus standard_space_roi for BET's -B option
  - Python packages: numpy, SimpleITK

Usage:
  python3 Pre_Processing_v2.0.py --input <folder with T1 scans> --out <output folder>
"""

import os
import re
import csv
import glob
import shutil
import logging
import argparse
import tempfile
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import SimpleITK as sitk

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_SUFFIX = "_processed_Bet.nii.gz"
MASK_DIR = "masks"
MASK_SUFFIX = "_processed_Bet_mask.nii.gz"
VOLUME_CSV = "brain_volumes.csv"
DEFAULT_NAME_PATTERN = r"^PREVENT_(?P<patient>\d{3})_(?P<timepoint>BL|\d+_yr)$"

def parse_args():
    parser = argparse.ArgumentParser(
        description="Background removal, N4 bias correction and BET skull stripping "
                    "of T1-weighted scans for DeepHarP, followed by brain volume measurement."
    )
    parser.add_argument("--input", required=True,
                        help="Folder containing T1 scans (.nii or .nii.gz, searched recursively)")
    parser.add_argument("--out", required=True,
                        help="Output folder for the skull-stripped scans and brain_volumes.csv")
    parser.add_argument("--threshold", type=float, default=120.0,
                        help="Voxels below this intensity are set to 0 before N4 "
                             "(default: 120; use 0 to turn off). Intensity scales differ "
                             "between scanners, so check this value for your data.")
    parser.add_argument("--bet-frac", type=float, default=0.20,
                        help="BET fractional intensity threshold, 0-1 (default: 0.20); "
                             "smaller values give larger brain outlines")
    parser.add_argument("--no-bet-B", action="store_true",
                        help="Do not use BET's -B option (bias field and neck cleanup)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of scans to process in parallel (default: 4)")
    parser.add_argument("--name-pattern", default=DEFAULT_NAME_PATTERN,
                        help="Regular expression that reads the patient ID and timepoint from "
                             "the scan name (file name without .nii/.nii.gz), using the named "
                             "groups 'patient' and 'timepoint'. Default: "
                             f"'{DEFAULT_NAME_PATTERN}' (PREVENT_001_BL, PREVENT_001_1_yr, ...)")
    return parser.parse_args()

def check_fsl(use_bet_B):
    fsldir = os.environ.get("FSLDIR")
    if fsldir:
        os.environ["PATH"] = os.path.join(fsldir, "bin") + os.pathsep + os.environ.get("PATH", "")

    if shutil.which("bet") is None:
        raise RuntimeError(
            "FSL 'bet' not found. Install FSL and make sure FSLDIR is set and "
            "$FSLDIR/bin is on your PATH."
        )
    if use_bet_B and shutil.which("standard_space_roi") is None:
        raise RuntimeError(
            "FSL 'standard_space_roi' (needed for BET -B) not found. "
            "Your FSL installation may be incomplete, or use --no-bet-B."
        )

def run_bet(in_path, out_path, frac, use_bet_B):
    # -m also saves the binary brain mask as <out>_mask.nii.gz; it does not
    # change the skull-stripped image itself.
    cmd = ["bet", in_path, out_path, "-f", str(frac), "-m"]
    if use_bet_B:
        cmd.append("-B")
    env = dict(os.environ, FSLOUTPUTTYPE="NIFTI_GZ")
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if r.returncode != 0 or not os.path.exists(out_path):
        raise RuntimeError(
            f"BET failed:\nCMD: {' '.join(cmd)}\nSTDERR:\n{r.stderr}\nSTDOUT:\n{r.stdout}"
        )

def process_scan(in_path, out_path, mask_path, threshold, bet_frac, use_bet_B):
    name = os.path.basename(in_path)

    if os.path.exists(out_path):
        logger.info(f"Skipping {name}: output already exists")
        return "skipped"

    tmp_dir = tempfile.mkdtemp(prefix="deepharp_preproc_")
    try:
        img = sitk.Cast(sitk.ReadImage(in_path), sitk.sitkFloat32)

        # 1. Background removal
        if threshold > 0:
            arr = sitk.GetArrayFromImage(img)
            arr[arr < threshold] = 0
            clean = sitk.GetImageFromArray(arr)
            clean.CopyInformation(img)
            img = clean

        # 2. N4 bias field correction
        corrected = sitk.N4BiasFieldCorrectionImageFilter().Execute(img)

        # 3. Skull stripping with BET
        tmp_in = os.path.join(tmp_dir, "n4.nii.gz")
        tmp_out = os.path.join(tmp_dir, "bet.nii.gz")
        tmp_mask = os.path.join(tmp_dir, "bet_mask.nii.gz")
        sitk.WriteImage(corrected, tmp_in)
        run_bet(tmp_in, tmp_out, bet_frac, use_bet_B)

        # Move the mask first, so a finished brain output always has its mask
        if os.path.exists(tmp_mask):
            shutil.move(tmp_mask, mask_path)
        shutil.move(tmp_out, out_path)
        logger.info(f"OK {name} -> {os.path.basename(out_path)}")
        return "success"

    except Exception as e:
        logger.error(f"FAIL {name}: {e}")
        return "error"

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

def scan_name(path):
    fname = os.path.basename(path)
    for ext in (".nii.gz", ".nii"):
        if fname.endswith(ext):
            return fname[: -len(ext)]
    return fname

def find_scans(input_dir, out_dir):
    files = set()
    for pattern in ("*.nii", "*.nii.gz"):
        files.update(glob.glob(os.path.join(input_dir, "**", pattern), recursive=True))

    # Ignore anything already inside the output folder
    out_prefix = os.path.join(out_dir, "")
    return sorted(f for f in files if not os.path.abspath(f).startswith(out_prefix))

def parse_patient_timepoint(name, name_pattern):
    """Read the patient ID and timepoint from a scan name."""
    m = name_pattern.search(name)
    if not m:
        logger.warning(
            f"Could not read patient ID and timepoint from '{name}' with --name-pattern; "
            "using the whole scan name as the patient ID and leaving the timepoint blank"
        )
        return name, ""
    return m.group("patient"), m.group("timepoint")

def timepoint_order(timepoint):
    """Sort key putting BL first, then N_yr by number of years, then anything else."""
    if timepoint.upper() == "BL":
        return 0
    m = re.match(r"^(\d+)_yr$", timepoint)
    return int(m.group(1)) if m else float("inf")

def volume_cc(image_path):
    """Volume of all non-zero voxels in an image, in cubic centimetres (cc = mL)."""
    img = sitk.ReadImage(image_path)
    n_voxels = int(np.count_nonzero(sitk.GetArrayFromImage(img)))
    voxel_mm3 = float(np.prod(img.GetSpacing()))
    return n_voxels * voxel_mm3 / 1000.0

def write_brain_volumes(out_dir, name_pattern):
    """
    Measure the brain volume of every skull-stripped scan in out_dir and write
    Patient ID, Timepoint and Brain Volume (cc) to brain_volumes.csv in out_dir.

    The volume is taken from the BET brain mask. Scans processed before masks were
    saved have none, so their volume is taken from the non-zero voxels of the
    skull-stripped scan instead, which can be lower because voxels inside the brain
    that fell below --threshold are also 0.
    """
    brains = sorted(glob.glob(os.path.join(out_dir, "*" + OUTPUT_SUFFIX)))
    rows = []

    for brain_path in brains:
        name = os.path.basename(brain_path)[: -len(OUTPUT_SUFFIX)]
        mask_path = os.path.join(out_dir, MASK_DIR, name + MASK_SUFFIX)
        patient, timepoint = parse_patient_timepoint(name, name_pattern)

        try:
            if os.path.exists(mask_path):
                vol = volume_cc(mask_path)
            else:
                logger.warning(
                    f"No BET mask for {name}; measuring non-zero voxels of the skull-stripped "
                    "scan instead, which may underestimate the volume. Delete its output and "
                    "re-run to create the mask."
                )
                vol = volume_cc(brain_path)
        except Exception as e:
            logger.error(f"Volume FAIL {name}: {e}")
            continue

        rows.append([patient, timepoint, f"{vol:.2f}"])

    rows.sort(key=lambda r: (r[0], timepoint_order(r[1]), r[1]))

    csv_path = os.path.join(out_dir, VOLUME_CSV)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Patient ID", "Timepoint", "Brain Volume (cc)"])
        writer.writerows(rows)

    logger.info(f"Brain volumes for {len(rows)} scans saved to {csv_path}")

def main():
    args = parse_args()
    input_dir = os.path.abspath(args.input)
    out_dir = os.path.abspath(args.out)
    use_bet_B = not args.no_bet_B

    name_pattern = re.compile(args.name_pattern)
    if not {"patient", "timepoint"} <= set(name_pattern.groupindex):
        raise ValueError("--name-pattern must contain the named groups "
                         "(?P<patient>...) and (?P<timepoint>...)")

    check_fsl(use_bet_B)
    os.makedirs(os.path.join(out_dir, MASK_DIR), exist_ok=True)

    scans = find_scans(input_dir, out_dir)
    logger.info(
        f"Found {len(scans)} scans in {input_dir} "
        f"(threshold={args.threshold}, bet_frac={args.bet_frac}, bet_B={use_bet_B})"
    )

    counts = {"success": 0, "skipped": 0, "error": 0}
    seen = set()

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = []
        for in_path in scans:
            name = scan_name(in_path)
            if name in seen:
                logger.warning(f"Duplicate scan name {name}, skipping {in_path}")
                counts["error"] += 1
                continue
            seen.add(name)

            out_path = os.path.join(out_dir, name + OUTPUT_SUFFIX)
            mask_path = os.path.join(out_dir, MASK_DIR, name + MASK_SUFFIX)
            futures.append(executor.submit(
                process_scan, in_path, out_path, mask_path,
                args.threshold, args.bet_frac, use_bet_B
            ))

        for fut in as_completed(futures):
            counts[fut.result()] += 1

    logger.info(
        f"Done. Processed: {counts['success']}, skipped: {counts['skipped']}, "
        f"errors: {counts['error']}"
    )
    logger.info(f"Skull-stripped scans saved to {out_dir}")

    write_brain_volumes(out_dir, name_pattern)


if __name__ == "__main__":
    main()

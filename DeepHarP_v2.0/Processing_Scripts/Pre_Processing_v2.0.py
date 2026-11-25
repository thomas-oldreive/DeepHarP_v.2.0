#!/usr/bin/env python3
"""
DeepHarP v2.0 - pre-processing of T1-weighted MRI scans.

For each scan, this script:
  1. Sets low-intensity background voxels to zero (see --threshold, adjust as needed).
  2. Corrects intensity inhomogeneity with N4 bias field correction (SimpleITK).
  3. Removes the skull with FSL BET.

Outputs are named <scan name>_processed_Bet.nii.gz, ready to place in the
DeepHarP Brains folder.

Requirements:
  - FSL on your PATH: bet, plus standard_space_roi for BET's -B option
  - Python packages: numpy, SimpleITK

Usage:
  python3 Pre_Processing_v.2.0.py --input <folder with T1 scans> --out <output folder>
"""

import os
import glob
import shutil
import logging
import argparse
import tempfile
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed

import SimpleITK as sitk

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_SUFFIX = "_processed_Bet.nii.gz"

def parse_args():
    parser = argparse.ArgumentParser(
        description="Background removal, N4 bias correction and BET skull stripping "
                    "of T1-weighted scans for DeepHarP."
    )
    parser.add_argument("--input", required=True,
                        help="Folder containing T1 scans (.nii or .nii.gz, searched recursively)")
    parser.add_argument("--out", required=True,
                        help="Output folder for the skull-stripped scans")
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
    cmd = ["bet", in_path, out_path, "-f", str(frac)]
    if use_bet_B:
        cmd.append("-B")
    env = dict(os.environ, FSLOUTPUTTYPE="NIFTI_GZ")
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if r.returncode != 0 or not os.path.exists(out_path):
        raise RuntimeError(
            f"BET failed:\nCMD: {' '.join(cmd)}\nSTDERR:\n{r.stderr}\nSTDOUT:\n{r.stdout}"
        )

def process_scan(in_path, out_path, threshold, bet_frac, use_bet_B):
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
        sitk.WriteImage(corrected, tmp_in)
        run_bet(tmp_in, tmp_out, bet_frac, use_bet_B)

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

def main():
    args = parse_args()
    input_dir = os.path.abspath(args.input)
    out_dir = os.path.abspath(args.out)
    use_bet_B = not args.no_bet_B

    check_fsl(use_bet_B)
    os.makedirs(out_dir, exist_ok=True)

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
            futures.append(executor.submit(
                process_scan, in_path, out_path, args.threshold, args.bet_frac, use_bet_B
            ))

        for fut in as_completed(futures):
            counts[fut.result()] += 1

    logger.info(
        f"Done. Processed: {counts['success']}, skipped: {counts['skipped']}, "
        f"errors: {counts['error']}"
    )
    logger.info(f"Skull-stripped scans saved to {out_dir}")


if __name__ == "__main__":
    main()
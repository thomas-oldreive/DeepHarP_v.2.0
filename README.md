# DeepHarP_v.2.0
This is a hippocampal segmentation tool that replicates manual hand tracing with higher accuracy than other automated segementation tools (see original DeepHarP publication). Intended to be used with T1-weighted MRI brain scans.

This is an updated version of DeepHarP_Hippocampus git repo, originally developed by Samaneh Nobakht, et al. published at: https://github.com/snobakht/DeepHarP_Hippocampus. When using this tool please cite the original DeepHarP paper (doi: 10.3390/s2107242), as well as the DeepMedic paper (doi: 10.1016/j.media.2016.10.004)

What's New in Version 2.0 (Updated fall 2026):
1) Updated syntax to python 3 vs. original python 2
2) Included suggested pre and post processing scripts for work to be more replicable
3) New README.md and Requirements.txt documentation
4) General cleaning and bug fixing

Requirements:
- Python 3.7
- Python packages: `pip install -r Requirements.txt`
- NifyReg (http://cmictig.cs.ucl.ac.uk/wiki/index.php/NiftyReg_install)

Usage:
1. Skull-strip your T1 scans; this is required. You can use the pre-processing script, which also applies N4 bias correction and a background threshold (turn the threshold off with `--threshold 0`):
   `python Processing_Scripts/Pre_Processing_v.2.0.py --input <folder with raw scans> --out <output folder>`
2. Place the skull-stripped scans (`.nii` or `.nii.gz`) in the `Brains` folder. It should contain only the scans you want to segment, so remove files from any earlier run first.
3. In `Scripts/main.sh`, set `root_dir` to the folder that contains the `DeepHarp` folder.
4. Optional: in `Configurations/testConfig.cfg`, change `sessionName` to name the output folder for this run (currently `Session_1`).
5. Run: `bash Scripts/main.sh`

Outputs:
- Segmentations: `predictions/<sessionName>/predictions`
- Cropped hippocampal ROIs and intermediate registration scans: `Brains`
- [OPTIONAL, but recommended]: Use post-processing script to remove non-hippocampal noise from segmentations and save them to a new, clean location.

# DeepHarP_v.2.0
This is an updated version of DeepHarP_Hippocampus git repo, originally developed by Samaneh Nobakht, et al. published at: https://github.com/snobakht/DeepHarP_Hippocampus. When using this tool please cite the original DeepHarP paper (doi: 10.3390/s2107242), as well as the DeepMedic paper (doi: 10.1016/j.media.2016.10.004)

This is a hippocampal segmentation tool that replicates manual hand tracing with higher accuracy than other automated segementation tools (see original DeepHarP publication). Intended to be used with T1-weighted MRI brain scans.

Requirements:
- Python 3.12+
- Python packages: `pip install -r requirements.txt`
- NifyReg (http://cmictig.cs.ucl.ac.uk/wiki/index.php/NiftyReg_install)
- DeepMedic (https://github.com/deepmedic/deepmedic?tab=readme-ov-file)

What's New in Version 2.0 (Updated fall 2026):
1) Updated syntax to python 3 vs. original python 2
2) Included suggested pre and post processing scripts for work to be more replicable
3) New README.md and Requirements.txt documentation
4) General cleaning and bug fixing

Usage:
1. [OPTIONAL, but recommended]: Use pre-processing script at your discretion to prepare brains for segmentation (background removal, N2 bias field correction, and BET skull stripping)
2. Place processed scans into the "Brains" folder
4. In `Scripts/main.sh`, set `root_dir` to your DeepHarP installation folder
5. In `Configurations/modelConfig.cfg`, set `folderForOutput` to your output folder inside `Predictions`.
6. In `Configurations/testConfig.cfg`:
   - set `folderForOutput` to the same output folder
   - set `cnnModelFilePath` to the full path of the model. The model from the original paper is: `Configurations/trainSessionWithValidTiny/ASHCNN.trainSessionWithValidTiny.final.2019-09-15.20.32.16.829653.model.ckpt`.
7. From the `Scripts` folder, run: `bash  source ./main.sh`

Outputs:
- Segmentations: `[output folder path]`
- Cropped hippocampal ROIs and intermediate registration scans: `Brains`
- [OPTIONAL, but recommended]: Use post-processing script to remove non-hippocampal noise from segmentations.

#!/bin/bash

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

echo "running gunzip 1"
cd "$root_dir/DeepHarp/Brains/"
gunzip *.nii.gz 2>/dev/null || true  # Ignore if no .gz

echo "running reorient.py"
python "$root_dir/DeepHarp/Scripts/reorient.py" "$root_dir/DeepHarp/Brains" "$root_dir/DeepHarp/Atlases/Reference.nii"

for i in *.nii; do
[ -e "$i" ] || continue  # skip if Brains has no .nii files

echo "running reg_aladin"
name="${i%.nii}"
reg_aladin -ref "$root_dir/DeepHarp/Atlases/MNI152_T1_1mm_brain.nii.gz" -flo "${name}.nii" -rigOnly -res "${name}_Aladin.nii"

echo "running reg_f3d"
reg_f3d -ref "${name}_Aladin.nii" -flo "$root_dir/DeepHarp/Atlases/MNI152_T1_1mm_brain.nii.gz" -res "${name}_regf3d_result.nii" -cpp "${name}_regf3d_cpp.nii"

echo "running reg_resample 1 (Right hippocampus) for $name"
reg_resample -ref "${name}_Aladin.nii" \
             -flo "$root_dir/DeepHarp/Atlases/RightHippocampusBinDil.nii.gz" \
             -trans "${name}_regf3d_cpp.nii" \
             -res "${name}_Right.nii" \
             -inter 0

if [ -f "${name}_Right.nii.gz" ]; then
        echo "Decompressing ${name}_Right.nii.gz"
        gunzip -f "${name}_Right.nii.gz"
fi
    
if [ ! -f "${name}_Right.nii" ]; then
    echo "ERROR: reg_resample failed to create ${name}_Right.nii"
    ls -lh "${name}"_Right* 2>/dev/null
else
    echo "SUCCESS: Created ${name}_Right.nii"
fi

echo "running reg_resample 2 (Left hippocampus) for $name"
reg_resample -ref "${name}_Aladin.nii" \
             -flo "$root_dir/DeepHarp/Atlases/LeftHippocampusBinDil.nii.gz" \
             -trans "${name}_regf3d_cpp.nii" \
             -res "${name}_Left.nii" \
             -inter 0
    

if [ -f "${name}_Left.nii.gz" ]; then
    echo "Decompressing ${name}_Left.nii.gz"
    gunzip -f "${name}_Left.nii.gz"
fi
    
if [ ! -f "${name}_Left.nii" ]; then
    echo "ERROR: reg_resample failed to create ${name}_Left.nii"
    ls -lh "${name}"_Left* 2>/dev/null
else
    echo "SUCCESS: Created ${name}_Left.nii"
fi

done

echo "running gunzip 2"
gunzip *.nii.gz 2>/dev/null || true  # Ignore if no .gz
for f in *_Aladin.nii; do
[ -e "$f" ] || continue

name="${f%_Aladin.nii*}"

echo "Running crop.py right"
python "$root_dir/DeepHarp/Scripts/crop.py" "${name}_Right.nii" "${name}_Aladin.nii" "${name}_croppedMask_Right.nii" "${name}_croppedROI_Right.nii"
echo "Running crop.py left"
python "$root_dir/DeepHarp/Scripts/crop.py" "${name}_Left.nii" "${name}_Aladin.nii" "${name}_croppedMask_Left.nii" "${name}_croppedROI_Left.nii"
echo "Running resamp.py right"
python "$root_dir/DeepHarp/Scripts/resamp.py" "${name}_croppedROI_Right.nii" "${name}_ResampRight.nii"
echo "Running resamp.py left"
python "$root_dir/DeepHarp/Scripts/resamp.py" "${name}_croppedROI_Left.nii" "${name}_ResampLeft.nii"
echo "Running norm.py right"
python "$root_dir/DeepHarp/Scripts/norm.py" "${name}_ResampRight.nii" "${name}_NormRight.nii"
echo "Running norm.py left"
python "$root_dir/DeepHarp/Scripts/norm.py" "${name}_ResampLeft.nii" "${name}_NormLeft.nii"

echo "Running flip.py right"
python "$root_dir/DeepHarp/Scripts/flip.py" "${name}_NormRight.nii" "${name}_croppedMask_Right.nii" "${name}_ROIFineRight.nii" "${name}_MaskFineRight.nii"
echo "Running flip.py left"
python "$root_dir/DeepHarp/Scripts/flip.py" "${name}_NormLeft.nii" "${name}_croppedMask_Left.nii" "${name}_ROIFineLeft.nii" "${name}_MaskFineLeft.nii"


done


echo "Running PathWriter.py right"
python "$root_dir/DeepHarp/Scripts/PathWriter.py" "$root_dir/DeepHarp/Brains/" "$root_dir/DeepHarp/Configurations/ROI.txt" "_ROIFineRight.nii"
echo "Running PathWriter.py left"
python "$root_dir/DeepHarp/Scripts/PathWriter.py" "$root_dir/DeepHarp/Brains/" "$root_dir/DeepHarp/Configurations/ROI.txt" "_ROIFineLeft.nii"

echo "Running PathMaskWriter.py"
python "$root_dir/DeepHarp/Scripts/PathMaskWriter.py" "$root_dir/DeepHarp/Configurations/ROI.txt" "$root_dir/DeepHarp/Configurations/Mask.txt"

echo "Running PredWriter.py right"
python "$root_dir/DeepHarp/Scripts/PredWriter.py" "$root_dir/DeepHarp/Brains/" "$root_dir/DeepHarp/Configurations/Predict.txt" "_ROIFineRight.nii"
echo "Running PredWriter.py left"
python "$root_dir/DeepHarp/Scripts/PredWriter.py" "$root_dir/DeepHarp/Brains/" "$root_dir/DeepHarp/Configurations/Predict.txt" "_ROIFineLeft.nii"
cd "$root_dir/DeepHarp/deepmedic-master"
###if gpu available at below add -dev cuda0

echo "Running DeepMedic prediction"
python ./deepMedicRun -test "$root_dir/DeepHarp/Configurations/testConfig.cfg" -model "$root_dir/DeepHarp/Configurations/modelConfig.cfg"

cd "$root_dir/DeepHarp/Configurations/"

rm Predict.txt Mask.txt ROI.txt

test_cfg="$root_dir/DeepHarp/Configurations/testConfig.cfg"
pred_dir=$(python - "$test_cfg" <<'EOF'
import os, sys
cfg_path = sys.argv[1]
cfg = {}
exec(open(cfg_path).read(), cfg)  # read the config the same way DeepMedic does
session = cfg.get("sessionName")
if session is None:
    session = "testSession"       # DeepMedic's default
out = cfg["folderForOutput"]
if not os.path.isabs(out):
    out = os.path.join(os.path.dirname(cfg_path), out)
print(os.path.normpath(os.path.join(out, "predictions", session, "predictions")))
EOF
)
echo "Predictions folder: $pred_dir"
cd "$pred_dir" && rm -f *ProbMapClass*

## Uncomment if you want to delete intermediate scans after running
#cd "$root_dir/DeepHarp/Brains/"

#rm *MaskFine*  *Resamp* *Norm* *Aladin*  *.txt  CroppedImageout.nii  *regf3d*  *croppedROI* *croppedMask* *_Left.nii  *_Right.nii
import nibabel as nib
import os
import glob

def reorient_to_ref(input_img_path, ref_img_path, output_img_path):
    ref_img = nib.load(ref_img_path)
    in_img = nib.load(input_img_path)

    ref_ornt = nib.orientations.aff2axcodes(ref_img.affine)
    in_ornt = nib.orientations.aff2axcodes(in_img.affine)

    ref_ornt_struct = nib.orientations.axcodes2ornt(ref_ornt)
    in_ornt_struct = nib.orientations.axcodes2ornt(in_ornt)
    transform = nib.orientations.ornt_transform(in_ornt_struct, ref_ornt_struct)

    data_reoriented = nib.orientations.apply_orientation(in_img.get_fdata(), transform)
    # Use reference affine and header, so overlays/segmentations align
    output_img = nib.Nifti1Image(data_reoriented, ref_img.affine, header=ref_img.header)
    nib.save(output_img, output_img_path)
    print(f"Reoriented: {output_img_path}")

def batch_reorient(folder, ref_img_path):
    nii_patterns = ["*.nii", "*.nii.gz"]
    files = []
    for pattern in nii_patterns:
        files.extend(glob.glob(os.path.join(folder, pattern)))
    for file_path in files:
        reorient_to_ref(file_path, ref_img_path, file_path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Batch reorient all NIfTI in a folder to match reference.")
    parser.add_argument("folder", help="Folder containing NIfTI files.")
    parser.add_argument("ref_img", help="Path to reference NIfTI image.")
    args = parser.parse_args()
    batch_reorient(args.folder, args.ref_img)
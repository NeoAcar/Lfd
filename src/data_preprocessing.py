"""
data_preprocessing.py

Functions to create small subsets of BRATS2021 and ACDC data, converting selected slices
into (coords, vals) NumPy arrays for INR training.

Usage:
    - Place raw BRATS NIfTI folders under data/Raw/BRATS2021/
      Each patient folder must contain FLAIR.nii.gz and segmentation.nii.gz.
    - Place raw ACDC DICOM folders under data/Raw/ACDC/ and labels.json in the same directory.
    - Then run this script directly (no arguments needed), and it will produce:
        data/Processed/BRATS_Subset/{Tumor,NoTumor}/…_coords.npy, …_vals.npy
        data/Processed/ACDC_Subset/{NOR,DCM,HCM,MINF,RV}/…_coords.npy, …_vals.npy
"""

import os
import random
import json
import numpy as np
import nibabel as nib
import pydicom

# -------------------------------------------------------------
# 1. BRATS Subset: select up to 100 tumor and 100 no-tumor slices
# -------------------------------------------------------------

random.seed(42)

def sample_brats_slices(patient_folder,patient, num_non_tumor=100, num_tumor=100):
    """
    Given a BRATS patient folder containing FLAIR.nii.gz and segmentation.nii.gz,
    return two lists of slice indices: tumor slices and no-tumor slices.
    """
    flair_path = os.path.join(patient_folder, patient+"_flair.nii.gz")
    seg_path   = os.path.join(patient_folder, patient+"_seg.nii.gz")
    flair = nib.load(flair_path).get_fdata().astype(np.float32)
    seg   = nib.load(seg_path).get_fdata().astype(np.int16)
    H, W, D = flair.shape


    tumor_idxs = []
    non_tumor_idxs = []
    for k in range(100,120):
        slice_mask = seg[:, :, k]
        if np.any(slice_mask > 0):
            tumor_idxs.append(k)
        else:
            non_tumor_idxs.append(k)

    # Sample up to the requested number (if fewer exist, return all)
    tumor_sample = random.sample(tumor_idxs, min(1, len(tumor_idxs)))
    non_tumor_sample = random.sample(non_tumor_idxs, min(1, len(non_tumor_idxs)))

    return tumor_sample, non_tumor_sample


def slice_to_coord_val(slice_img, output_dir, prefix):
    """
    Given a 2D NumPy array 'slice_img' (H x W), normalize it to [0,1],
    build (coords, vals) arrays, and save them under output_dir with the given prefix.
    """
    H, W = slice_img.shape
    # Min-max normalization to [0,1]
    img_min, img_max = slice_img.min(), slice_img.max()
    norm_img = (slice_img - img_min) / (img_max - img_min + 1e-8)

    coords = []
    vals = []
    for i in range(H):
        for j in range(W):
            x = i / (H - 1)
            y = j / (W - 1)
            coords.append([x, y])
            vals.append(norm_img[i, j])

    coords = np.array(coords, dtype=np.float32)  # (H*W, 2)
    vals = np.array(vals, dtype=np.float32)      # (H*W,)

    coord_path = os.path.join(output_dir, f"{prefix}_coords.npy")
    vals_path  = os.path.join(output_dir, f"{prefix}_vals.npy")
    np.save(coord_path, coords)
    np.save(vals_path, vals)


def process_brats_subset(raw_root="data/Raw/BRATS2021", processed_root="data/Processed/BRATS_Subset",
                         max_tumor=100, max_no_tumor=100):
    """
    Iterate over all patient folders under raw_root, sample up to max_tumor tumor slices
    and max_no_tumor no-tumor slices in total (across all patients), then convert each
    selected slice to (coords, vals) and save under processed_root/Tumor/ or processed_root/NoTumor/.
    """
    # Gather all patient folders
    patients = sorted([d for d in os.listdir(raw_root) if os.path.isdir(os.path.join(raw_root, d))])
    tumor_count = 0
    non_tumor_count = 0

    # Ensure output directories exist
    tumor_out = os.path.join(processed_root, "Tumor")
    no_tumor_out = os.path.join(processed_root, "NoTumor")
    os.makedirs(tumor_out, exist_ok=True)
    os.makedirs(no_tumor_out, exist_ok=True)

    for patient in patients:
        patient_folder = os.path.join(raw_root, patient)
        tumor_idxs, no_tumor_idxs = sample_brats_slices(patient_folder,patient,
                                                       num_non_tumor=max_no_tumor - non_tumor_count,
                                                       num_tumor=max_tumor - tumor_count)

        flair = nib.load(os.path.join(patient_folder, patient+"_flair.nii.gz")).get_fdata().astype(np.float32)

        # Process tumor slices
        for k in tumor_idxs:
            if tumor_count >= max_tumor:
                break
            slice_img = flair[:, :, k]
            prefix = f"{patient}_slice_{k:03d}"
            slice_to_coord_val(slice_img, tumor_out, prefix)
            tumor_count += 1

        # Process no-tumor slices
        for k in no_tumor_idxs:
            if non_tumor_count >= max_no_tumor:
                break
            slice_img = flair[:, :, k]
            prefix = f"{patient}_slice_{k:03d}"
            slice_to_coord_val(slice_img, no_tumor_out, prefix)
            non_tumor_count += 1

        # Stop early if both quotas are met
        if tumor_count >= max_tumor and non_tumor_count >= max_no_tumor:
            break

    print(f"BRATS subset complete: {tumor_count} tumor slices, {non_tumor_count} no-tumor slices saved.")


# -------------------------------------------------------------
# 2. ACDC Subset: select up to 40 slices per class (5 classes)
# -------------------------------------------------------------

def sample_acdc_slices(raw_root="data/Raw/ACDC", labels_path="data/Raw/ACDC/labels.json",
                       max_per_class=40):
    """
    Read labels.json mapping patient_id -> class label (NOR, DCM, HCM, MINF, RV).
    For each class, randomly sample up to max_per_class slices across all patients.
    Return a dict: {class_label: [(patient_id, filename), ...]}.
    """
    with open(labels_path, 'r') as f:
        labels_dict = json.load(f)  # { "patient_0001": "HCM", ... }

    class_to_slices = {cls: [] for cls in ["NOR", "DCM", "HCM", "MINF", "RV"]}
    for patient_id, cls in labels_dict.items():
        patient_folder = os.path.join(raw_root, patient_id)
        if not os.path.isdir(patient_folder):
            continue
        for fname in sorted(os.listdir(patient_folder)):
            if fname.endswith(".dcm"):
                class_to_slices[cls].append((patient_id, fname))

    sampled = {}
    for cls, slice_list in class_to_slices.items():
        sampled[cls] = random.sample(slice_list, min(max_per_class, len(slice_list)))

    return sampled


def process_acdc_subset(raw_root="data/Raw/ACDC", labels_path="data/Raw/ACDC/labels.json",
                        processed_root="data/Processed/ACDC_Subset", max_per_class=40):
    """
    Given raw_root and labels.json, sample up to max_per_class slices per class and
    convert each to (coords, vals). Save results under processed_root/{class}/.
    """
    sampled_dict = sample_acdc_slices(raw_root, labels_path, max_per_class)

    for cls, slice_infos in sampled_dict.items():
        out_dir = os.path.join(processed_root, cls)
        os.makedirs(out_dir, exist_ok=True)
        for (patient_id, fname) in slice_infos:
            dicom_path = os.path.join(raw_root, patient_id, fname)
            ds = pydicom.dcmread(dicom_path)
            img = ds.pixel_array.astype(np.float32)
            img_min, img_max = img.min(), img.max()
            img_norm = (img - img_min) / (img_max - img_min + 1e-8)
            slice_to_coord_val(img_norm, out_dir, f"{patient_id}_{fname.split('.')[0]}")
    print("ACDC subset complete: up to 40 slices per class saved.")


# -------------------------------------------------------------
# 3. Helper: Load full-volume functions (if needed elsewhere)
# -------------------------------------------------------------

def load_brats_flair(patient_folder):
    """
    Load the 3D FLAIR volume as a NumPy array normalized to [0,1].
    Returns: img_data shape = (H, W, D)
    """
    flair_path = os.path.join(patient_folder, "FLAIR.nii.gz")
    img_obj = nib.load(flair_path)
    img_data = img_obj.get_fdata().astype(np.float32)
    img_min, img_max = img_data.min(), img_data.max()
    img_data = (img_data - img_min) / (img_max - img_min + 1e-8)
    return img_data


def load_acdc_slices(patient_folder):
    """
    Load all DICOM slices in a folder as a NumPy array of shape (num_slices, H, W),
    normalized to [0,1].
    """
    slice_list = []
    for fname in sorted(os.listdir(patient_folder)):
        if fname.endswith(".dcm"):
            ds = pydicom.dcmread(os.path.join(patient_folder, fname))
            img = ds.pixel_array.astype(np.float32)
            img_min, img_max = img.min(), img.max()
            img_norm = (img - img_min) / (img_max - img_min + 1e-8)
            slice_list.append(img_norm)
    if len(slice_list) == 0:
        return None
    return np.stack(slice_list, axis=0)  # shape = (num_slices, H, W)


# -------------------------------------------------------------
# 4. Main entrypoint: generate both subsets
# -------------------------------------------------------------

if __name__ == "__main__":
    # 1) Process BRATS subset
    print("Processing BRATS2021 subset...")
    process_brats_subset(
        raw_root="data/Raw/BRATS2021",
        processed_root="data/Processed/BRATS_Subset",
        max_tumor=100,
        max_no_tumor=100
    )
    # 2) Process ACDC subset
    # print("Processing ACDC subset...")
    # process_acdc_subset(
    #     raw_root="data/Raw/ACDC",
    #     labels_path="data/Raw/ACDC/labels.json",
    #     processed_root="data/Processed/ACDC_Subset",
    #     max_per_class=40
    # )

    print("All preprocessing complete.")

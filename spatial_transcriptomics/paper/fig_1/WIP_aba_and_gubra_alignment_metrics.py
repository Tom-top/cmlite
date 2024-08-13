import os

import numpy as np
import tifffile
import SimpleITK as sitk

import spatial_transcriptomics.utils.utils as ut


def dice_coefficient(segmentation1, segmentation2):
    """
    Computes the Dice Similarity Coefficient between two binary segmentation masks.

    Parameters:
    segmentation1 (numpy array): Binary mask of the first segmentation.
    segmentation2 (numpy array): Binary mask of the second segmentation.

    Returns:
    float: Dice Similarity Coefficient.
    """
    # Flatten the arrays to make sure they are one-dimensional
    segmentation1 = segmentation1.flatten()
    segmentation2 = segmentation2.flatten()

    # Compute the intersection
    intersection = np.sum(segmentation1 * segmentation2)

    # Compute the Dice coefficient
    dice = (2. * intersection) / (np.sum(segmentation1) + np.sum(segmentation2))

    return dice


atlas_path = r"C:\Users\MANDOUDNA\PycharmProjects\cmlite\resources\atlas"
analyze = "aba_to_gubra_atlas"
analyze_split = analyze.split("_")
transformed_annotation = analyze_split[0]
target_annotation = analyze_split[2]
target_annotation_path = os.path.join(atlas_path, f"{target_annotation}_annotation_mouse.tif")
target_annotation_array = np.swapaxes(tifffile.imread(target_annotation_path), 0, 2)[::-1]
transformed_annotation_image = sitk.ReadImage(os.path.join(atlas_path, fr"atlas_transformations\{analyze}\result.mhd"))
transformed_annotation_array = sitk.GetArrayFromImage(transformed_annotation_image)

tifffile.imwrite(os.path.join(atlas_path, "target_atlas.tif"), target_annotation_array)
tifffile.imwrite(os.path.join(atlas_path, "transformed_atlas.tif"), transformed_annotation_array)

target_atlas_metadata = ut.load_json_file(os.path.join(atlas_path, f"{target_annotation}_annotation_mouse.json"))["msg"][0]
transformed_atlas_metadata = ut.load_json_file(os.path.join(atlas_path, f"{transformed_annotation}_annotation_mouse.json"))["msg"][0]
unique_ids = np.unique(target_annotation_array)
DSC_score = []
acros = []

for target_uid in unique_ids:
    target_metadata = ut.find_dict_by_key_value(target_atlas_metadata, target_uid, key="id")
    target_acro = target_metadata["acronym"]
    if target_acro not in ["universe"]:
        transformed_metadata = ut.find_dict_by_key_value(transformed_atlas_metadata, target_acro, key="acronym")
        transformed_uid = transformed_metadata["id"]
        target_mask = target_annotation_array == target_uid
        transformed_mask = transformed_annotation_array == transformed_uid
        # Compute adapted rand error which gives precision, recall, and F1-score (Dice coefficient)
        dice = dice_coefficient(target_mask, transformed_mask)
        print(f"Computing dice-sørensen score for {target_acro}! value: {dice}")
        DSC_score.append(dice)
        acros.append(target_acro)

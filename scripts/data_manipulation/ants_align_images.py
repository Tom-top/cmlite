"""
Script that aligns/transforms 3D images using ANTs (antspyx lib). ANTs uses a combination of Affine and
Symmetric Normalization for non-linear registration.
"""

import os

import numpy as np
import nibabel as nib
import tifffile
import ants

import utils.utils as ut

import resampling.resampling as res

resources_directory = r"C:\Users\MANDOUDNA\PycharmProjects\cmlite\resources\atlas"
working_directory = r"E:\LLL\MBP_data\raw\ID523_an000003_g001_Brain_M3_rescan3\xy5p0_z10p0\2024-08-30_231537_merged"
saving_directory = ut.create_dir(os.path.join(working_directory, "ants_output"))
default_syn_params = True

# ########################################################################################################################
# # ALIGN AUTO TO REFERENCE (25 um)
# ########################################################################################################################
#
# fixed_image_name = "gubra_reference_mouse_25um_-1_2_3_None"
# moving_image_name = "resampled_25um_7"
# saving_subdirectory = ut.create_dir(os.path.join(saving_directory, "auto_to_reference_25um"))
#
# # CONVERT FIXED IMAGE
# fixed_image_tif_path = os.path.join(resources_directory, f'{fixed_image_name}.tif')
# fixed_image_tif = tifffile.imread(fixed_image_tif_path)
# nifti_image = nib.Nifti1Image(fixed_image_tif.astype(np.float32), np.eye(4))
# fixed_image_nii_path = os.path.join(resources_directory, f'{fixed_image_name}.nii')
# nib.save(nifti_image, fixed_image_nii_path)
#
# # CONVERT MOVING IMAGE
# moving_image_tif_path = os.path.join(working_directory, f'{moving_image_name}.tif')
# moving_image_tif = tifffile.imread(moving_image_tif_path)
# nifti_image = nib.Nifti1Image(moving_image_tif.astype(np.float32), np.eye(4))
# moving_image_nii_path = os.path.join(working_directory, f'{moving_image_name}.nii')
# nib.save(nifti_image, moving_image_nii_path)
#
# fixed_image = ants.image_read(fixed_image_nii_path)
# moving_image = ants.image_read(moving_image_nii_path)
#
# # Step 1: Perform an initial affine registration and save the result
# affine_output_prefix = os.path.join(saving_subdirectory, 'affine')
# affine_registration = ants.registration(fixed=fixed_image, moving=moving_image, type_of_transform='Affine',
#                                         outprefix=affine_output_prefix, verbose=True)
# affine_aligned = affine_registration['warpedmovout']
# ants.image_write(affine_aligned, os.path.join(saving_subdirectory, 'affine_ants_output.nii'))
#
# # Step 2: Perform a SyN transformation using the affine-aligned image
# syn_output_prefix = os.path.join(saving_subdirectory, 'syn')
# # Perform SyN transformation with custom parameters
# if default_syn_params:
#     syn_registration = ants.registration(fixed=fixed_image, moving=affine_aligned, type_of_transform='SyN',
#                                          outprefix=syn_output_prefix, verbose=True)
# else:
#     syn_registration = ants.registration(
#         fixed=fixed_image,
#         moving=affine_aligned,
#         type_of_transform='SyN',
#         outprefix=syn_output_prefix,
#         reg_iterations=(100, 70, 50),  # Increase iterations for a more refined transformation
#         flow_sigma=0.5,  # Lower for finer deformation but be cautious of noise
#         total_sigma=0.5,  # Controls smoothness, adjust as necessary
#         grad_step=0.8,  # More aggressive gradient step
#         verbose=True
#     )
# syn_aligned = syn_registration['warpedmovout']
# ants.image_write(syn_aligned, os.path.join(saving_subdirectory, 'syn_ants_output.nii'))
#
# ########################################################################################################################
# # TRANSFORM SIGNAL TO REFERENCE (10 um)
# ########################################################################################################################
#
# signal_channels = [1, 3, 5]
#
# for signal_channel in signal_channels:
#
#     image_to_transform_name = f"resampled_10um_{signal_channel}"
#
#     # CONVERT SIGNAL IMAGE
#     image_to_transform_tif_path = os.path.join(working_directory, f'{image_to_transform_name}.tif')
#     image_to_transform_tif = tifffile.imread(image_to_transform_tif_path)
#     nifti_image = nib.Nifti1Image(image_to_transform_tif.astype(np.float32), np.eye(4))
#     image_to_transform_nii_path = os.path.join(working_directory, f'{image_to_transform_name}.nii')
#     nib.save(nifti_image, image_to_transform_nii_path)
#     image_to_transform = ants.image_read(image_to_transform_nii_path)
#
#     # Load transformation files
#     affine_transform = affine_output_prefix + '0GenericAffine.mat'
#     syn_transform = syn_output_prefix + '1Warp.nii.gz'
#     # syn_inverse = syn_output_prefix + '1InverseWarp.nii.gz'
#
#     # Apply transformations to the new image
#     transformed_image = ants.apply_transforms(fixed=fixed_image, moving=image_to_transform,
#                                               transformlist=[syn_transform, affine_transform])
#     ants.image_write(transformed_image, os.path.join(saving_subdirectory, f'{image_to_transform_name}_transformed.nii'))

########################################################################################################################
# ALIGN REFERENCE TO AUTO (25 um)
########################################################################################################################

fixed_image_name = "resampled_25um_7"
moving_image_name = "gubra_reference_mouse_-1_2_3_None"
saving_subdirectory = ut.create_dir(os.path.join(saving_directory, "reference_to_auto_25um"))

# CONVERT FIXED IMAGE
fixed_image_tif_path = os.path.join(working_directory, f'{fixed_image_name}.tif')
fixed_image_tif = tifffile.imread(fixed_image_tif_path)
nifti_image = nib.Nifti1Image(fixed_image_tif.astype(np.float32), np.eye(4))
fixed_image_nii_path = os.path.join(working_directory, f'{fixed_image_name}.nii')
nib.save(nifti_image, fixed_image_nii_path)

# CONVERT MOVING IMAGE
moving_image_tif_path = os.path.join(resources_directory, f'{moving_image_name}.tif')
moving_image_tif = tifffile.imread(moving_image_tif_path)
nifti_image = nib.Nifti1Image(moving_image_tif.astype(np.float32), np.eye(4))
moving_image_nii_path = os.path.join(working_directory, f'{moving_image_name}.nii')
nib.save(nifti_image, moving_image_nii_path)

fixed_image = ants.image_read(fixed_image_nii_path)
moving_image = ants.image_read(moving_image_nii_path)

# Step 1: Perform an initial affine registration and save the result
affine_output_prefix = os.path.join(saving_subdirectory, 'affine')
affine_registration = ants.registration(fixed=fixed_image, moving=moving_image, type_of_transform='Affine',
                                        outprefix=affine_output_prefix, verbose=True)
affine_aligned = affine_registration['warpedmovout']
ants.image_write(affine_aligned, os.path.join(saving_subdirectory, 'affine_ants_output.nii'))

# Step 2: Perform a SyN transformation using the affine-aligned image
syn_output_prefix = os.path.join(saving_subdirectory, 'syn')
# Perform SyN transformation with custom parameters
if default_syn_params:
    syn_registration = ants.registration(fixed=fixed_image, moving=affine_aligned, type_of_transform='SyN',
                                         outprefix=syn_output_prefix, verbose=True)
else:
    syn_registration = ants.registration(
        fixed=fixed_image,
        moving=affine_aligned,
        type_of_transform='SyN',
        outprefix=syn_output_prefix,
        reg_iterations=(100, 70, 50),  # Increase iterations for a more refined transformation
        flow_sigma=0.5,  # Lower for finer deformation but be cautious of noise
        total_sigma=0.5,  # Controls smoothness, adjust as necessary
        grad_step=0.8,  # More aggressive gradient step
        verbose=True
    )
syn_aligned = syn_registration['warpedmovout']
ants.image_write(syn_aligned, os.path.join(saving_subdirectory, 'syn_ants_output.nii'))

########################################################################################################################
# TRANSFORM ATLAS TO AUTO (25 um)
########################################################################################################################

image_to_transform_name = "gubra_annotation_mouse_-1_2_3_None"

# CONVERT SIGNAL IMAGE
image_to_transform_tif_path = os.path.join(resources_directory, f'{image_to_transform_name}.tif')
image_to_transform_tif = tifffile.imread(image_to_transform_tif_path)
nifti_image = nib.Nifti1Image(image_to_transform_tif.astype(np.float32), np.eye(4))
image_to_transform_nii_path = os.path.join(working_directory, f'{image_to_transform_name}.nii')
nib.save(nifti_image, image_to_transform_nii_path)
image_to_transform = ants.image_read(image_to_transform_nii_path)

# Load transformation files
affine_transform = affine_output_prefix + '0GenericAffine.mat'
syn_transform = syn_output_prefix + '1Warp.nii.gz'
# syn_inverse = syn_output_prefix + '1InverseWarp.nii.gz'

# Apply transformations to the new image
transformed_image = ants.apply_transforms(fixed=fixed_image, moving=image_to_transform,
                                          transformlist=[syn_transform, affine_transform],
                                          interpolator="nearestNeighbor")
ants.image_write(transformed_image, os.path.join(saving_subdirectory, f'transformed_annotation.nii'))

########################################################################################################################
# UPSAMPLE TRANSFORMED ATLAS
########################################################################################################################

resampled_channel = tifffile.imread(os.path.join(working_directory, f"resampled_10um_3.tif"))
resampled_channel = np.swapaxes(resampled_channel, 0, 2)

atlas_path = os.path.join(saving_subdirectory, "transformed_annotation.nii")
transformed_atlas = nib.load(atlas_path).get_fdata()
transformed_atlas = np.array(np.swapaxes(transformed_atlas, 0, 2))
atlas_10um_path = os.path.join(saving_subdirectory, "transformed_annotation_10um.tif")

resample_10um_parameter = {
    "source_resolution": (25, 25, 25),
    "sink_resolution": (10, 10, 10),
    "sink_shape": resampled_channel.shape,
    "processes": None,
    "verbose": True,
    "method": "memmap",
    "interpolation": 'nearest'
}

res.resample(transformed_atlas, sink=atlas_10um_path, **resample_10um_parameter)

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

resources_directory = r"C:\Users\MANDOUDNA\PycharmProjects\cmlite\resources\atlas"
working_directory = r"E:\tto\test"
saving_directory = ut.create_dir(os.path.join(working_directory, "ants_output"))
default_syn_params = True

########################################################################################################################
# ALIGN GUBRA REFERENCE TO ABA REFERENCE
########################################################################################################################

fixed_image_name = "aba_reference_mouse"
moving_image_name = "gubra_reference_mouse"
saving_subdirectory = ut.create_dir(os.path.join(saving_directory, "gubra_to_aba"))

# CONVERT FIXED IMAGE
fixed_image_tif_path = os.path.join(resources_directory, f'{fixed_image_name}.tif')
fixed_image_tif = tifffile.imread(fixed_image_tif_path)
nifti_image = nib.Nifti1Image(fixed_image_tif.astype(np.float32), np.eye(4))
fixed_image_nii_path = os.path.join(saving_subdirectory, f'{fixed_image_name}.nii')
nib.save(nifti_image, fixed_image_nii_path)

# CONVERT MOVING IMAGE
moving_image_tif_path = os.path.join(resources_directory, f'{moving_image_name}.tif')
moving_image_tif = tifffile.imread(moving_image_tif_path)
nifti_image = nib.Nifti1Image(moving_image_tif.astype(np.float32), np.eye(4))
moving_image_nii_path = os.path.join(saving_subdirectory, f'{moving_image_name}.nii')
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
        reg_iterations=(50, 40, 30),  # Fewer iterations for a softer transformation
        flow_sigma=2,  # Lower flow_sigma for stronger deformation field
        total_sigma=0,  # Adjust to control smoothness of final deformation
        grad_step=0.21,  # Increase gradient step for more aggressive transformation
        outprefix=syn_output_prefix,
        verbose=True
    )
syn_aligned = syn_registration['warpedmovout']
ants.image_write(syn_aligned, os.path.join(saving_subdirectory, 'syn_ants_output.nii'))

########################################################################################################################
# TRANSFORM GUBRA ATLAS TO ABA REFERENCE
########################################################################################################################

image_to_transform_name = f"gubra_annotation_mouse"

# CONVERT SIGNAL IMAGE
image_to_transform_tif_path = os.path.join(resources_directory, f'{image_to_transform_name}.tif')
image_to_transform_tif = tifffile.imread(image_to_transform_tif_path)
nifti_image = nib.Nifti1Image(image_to_transform_tif.astype(np.float32), np.eye(4))
image_to_transform_nii_path = os.path.join(saving_subdirectory, f'{image_to_transform_name}.nii')
nib.save(nifti_image, image_to_transform_nii_path)
image_to_transform = ants.image_read(image_to_transform_nii_path)

# Load transformation files
affine_transform = affine_output_prefix + '0GenericAffine.mat'
syn_transform = syn_output_prefix + '1Warp.nii.gz'
# syn_inverse = syn_output_prefix + '1InverseWarp.nii.gz'

# Apply transformations to the new image
transformed_image = ants.apply_transforms(fixed=fixed_image,
                                          moving=image_to_transform,
                                          transformlist=[syn_transform, affine_transform],
                                          interpolator='nearestNeighbor',  # Avoids interpolation
                                          )
ants.image_write(transformed_image, os.path.join(saving_subdirectory, f'{image_to_transform_name}_transformed.nii'))

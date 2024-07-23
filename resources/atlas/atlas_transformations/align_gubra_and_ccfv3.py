import os

import tifffile

from ClearMap.Environment import *

working_directory = r"U:\Users\TTO\spatial_transcriptomics\atlas_ressources"
moving_image = os.path.join(working_directory, "gubra_template_coronal.tif")
fixed_image = os.path.join(working_directory, "ABA_25um_reference_coronal.tif")

resources_directory = settings.resources_path
align_reference_affine_file = io.join(resources_directory, 'Alignment/align_affine.txt')
align_reference_bspline_file = io.join(resources_directory, 'Alignment/align_bspline.txt')

# align Gubra atlas to ABA
align_reference_parameter = {
    "moving_image": moving_image,
    "fixed_image": fixed_image,
    "affine_parameter_file": align_reference_affine_file,
    "bspline_parameter_file": align_reference_bspline_file,
    "result_directory": os.path.join(working_directory, "gubra_to_aba"),
}

elx.align(**align_reference_parameter)


# align ABA atlas to Gubra
align_reference_parameter = {
    "moving_image": fixed_image,
    "fixed_image": moving_image,
    "affine_parameter_file": align_reference_affine_file,
    "bspline_parameter_file": align_reference_bspline_file,
    "result_directory": os.path.join(working_directory, "aba_to_gubra"),
}

elx.align(**align_reference_parameter)
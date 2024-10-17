import os, shutil
import json
import requests

import numpy as np
import pandas as pd
import anndata
import tifffile

# import tifffile

import utils.utils as ut

import alignment.align as elx

transform_directory = r"C:\Users\MANDOUDNA\PycharmProjects\cmlite\resources\atlas\atlas_transformations\transform_files_elastix"
result_directory = os.path.join(r"E:\tto\mapping_aba_to_gubra", "transformed_reference")

transform_atlas_parameter = {'source': r"E:\tto\mapping_aba_to_gubra\gubra_reference_mouse.tif",
                             'result_directory': result_directory,
                             'transform_parameter_file': os.path.join(transform_directory, "lsfm_2_ccfv3_transform_tif_2.txt")
                             }

if os.path.exists(result_directory):
    shutil.rmtree(result_directory)

elx.transform_images(**transform_atlas_parameter)



# import nibabel as nib
#
# test = nib.load(r"C:\Users\MANDOUDNA\PycharmProjects\cmlite\resources\atlas\atlas_transformations\deformation_fields\lsfm_2_ccfv3_deffield.nii.gz")
# test_data = test.get_fdata()
# test_data = test_data.astype("uint32")
# test_data = np.squeeze(test_data, axis=3)
# test_data = np.swapaxes(test_data, 0, 2)
# test_data = np.flip(test_data, 0)
# # os.remove(os.path.join(r"E:\tto\mapping_aba_to_gubra", "test_deformation_field.tif"))
# tifffile.imwrite(os.path.join(r"E:\tto\mapping_aba_to_gubra", "test_deformation_field.tif"), test_data)

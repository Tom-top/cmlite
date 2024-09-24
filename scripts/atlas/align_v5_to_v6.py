import os

from natsort import natsorted
import tifffile
import nibabel as nib
import numpy as np

import alignment.align as elx

working_directory = r"E:\tto\SfN\Henrik"
average_directory = os.path.join(working_directory, "GUI36-2021-433_g001_average")
transformation_file_directory = r"C:\Users\MANDOUDNA\PycharmProjects\cmlite\resources\atlas\atlas_transformations" \
                                r"\transform_files_elastix"

data = [tifffile.imread(os.path.join(average_directory, i)) for i in natsorted(os.listdir(average_directory))]
data = np.array(data)

def transform_v5_to_v6(result_directory, img_v5, verbose=False):
    if type(img_v5) == np.ndarray:
        img_v5_data = np.swapaxes(img_v5, 0, 2)
    else:
        img_v5_data = img_v5.get_fdata()
    file_transform = os.path.join(transformation_file_directory, "gubra_template_v5_2_v6_deffield_transform.txt")
    img_v5_pad = np.pad(img_v5_data, ((0, 0), (145, 0), (0, 0)))  # add padding
    file_img_moving = os.path.join(result_directory, 'original.nii.gz')
    affine = [[-1., 0., 0., -230.],
              [0., -1., 0., -310.],
              [0., 0., 1., -161.],
              [0., 0., 0., 1.]]
    nib.save(nib.Nifti1Image(img_v5_pad, affine), file_img_moving)

    # construct transformix call
    program = r'C:\Users\MANDOUDNA\PycharmProjects\cmlite\external\elastix\windows\transformix'
    param_in = ' -in ' + file_img_moving
    param_out = ' -out ' + result_directory
    param_tp = ' -tp ' + file_transform
    param_verbose = ''

    # small hack to verbosity as it doesn't have an argument in transformix
    if not verbose:
        file_verbose = os.path.join(result_directory, 'verbose.txt')
        param_verbose = ' > ' + file_verbose

    print(program + param_in + param_out + param_tp + param_verbose)
    os.system(program + param_in + param_out + param_tp + param_verbose)

v5_to_v6_directory = os.path.join(working_directory, f"v5_to_v6")
transform_v5_to_v6(v5_to_v6_directory, data, verbose=False)

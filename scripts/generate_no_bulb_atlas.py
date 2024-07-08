import os

import tifffile

atlas_path = r"gubra_reference_mouse.tif"
atlas = tifffile.imread(atlas_path)

atlas[:, :100, :] = 0

tifffile.imwrite(r"gubra_reference_nb_mouse.tif",
                 atlas)
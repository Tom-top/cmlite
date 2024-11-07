import os

import alignment.align as elx

sample_directory = r"U:\Users\TTO\spatial_transcriptomics\transform_points_to_gubra"
fixed_image_path = os.path.join(sample_directory, "ccfv3_to_gubra.tif")
moving_image_path = os.path.join(sample_directory, "ccfv3.tif")

aba_to_gubra_directory = os.path.join(sample_directory, f"ccfv3_to_gubra")
align_auto_to_reference = dict(fixed_image_path=fixed_image_path,
                               moving_image_path=moving_image_path,
                               affine_parameter_file="resources/alignment/align_affine.txt",
                               bspline_parameter_file="resources/alignment/align_bspline.txt",
                               output_dir=aba_to_gubra_directory,
                               )
elx.align_images(**align_auto_to_reference)

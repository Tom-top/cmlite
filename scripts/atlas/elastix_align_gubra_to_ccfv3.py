import os

import alignment.align as elx

sample_directory = r"U:\Users\TTO\spatial_transcriptomics\transform_points_to_gubra"
fixed_image_path = os.path.join(sample_directory, "gubra_to_ccfv3.tif")
moving_image_path = os.path.join(sample_directory, "gubra.tif")

gubra_to_aba_directory = os.path.join(sample_directory, f"gubra_to_aba")
align_auto_to_reference = dict(fixed_image_path=fixed_image_path,
                               moving_image_path=moving_image_path,
                               affine_parameter_file="resources/alignment/align_affine.txt",
                               bspline_parameter_file="resources/alignment/align_bspline.txt",
                               output_dir=gubra_to_aba_directory,
                               )
elx.align_images(**align_auto_to_reference)

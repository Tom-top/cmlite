import SimpleITK as sitk


# def align_images(fixed_image_path, moving_image_path, parameters_path, output_dir, output_file_name,
#                  transform_file_name):
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#
#     # Read images
#     fixed_image = sitk.ReadImage(fixed_image_path, sitk.sitkFloat32)
#     moving_image = sitk.ReadImage(moving_image_path, sitk.sitkFloat32)
#
#     # Setup elastix image filter
#     elastix_image_filter = sitk.ElastixImageFilter()
#     elastix_image_filter.SetFixedImage(fixed_image)
#     elastix_image_filter.SetMovingImage(moving_image)
#
#     # Read parameter map from file
#     parameter_map = sitk.ReadParameterFile(parameters_path)
#
#     # Set the output directory for elastix log files
#     log_output_dir = os.path.join(output_dir, 'elastix_logs')
#     if not os.path.exists(log_output_dir):
#         os.makedirs(log_output_dir)
#
#     # Set additional parameters in the parameter map
#     parameter_map['WriteLogFile'] = ['true']
#     parameter_map['WriteResultImageAfterEachResolution'] = ['false']
#     parameter_map['WriteResultImage'] = ['false']
#
#     elastix_image_filter.SetParameterMap(parameter_map)
#
#     # Temporarily change the working directory to the log output directory
#     original_cwd = os.getcwd()
#     os.chdir(log_output_dir)
#
#     try:
#         # Perform registration
#         elastix_image_filter.Execute()
#     finally:
#         # Revert to the original working directory
#         os.chdir(original_cwd)
#
#     # Get the result image
#     result_image = elastix_image_filter.GetResultImage()
#
#     # Save the result image
#     sitk.WriteImage(result_image, os.path.join(output_dir, output_file_name))
#
#     # Get and save the transformation parameters
#     transform_parameter_map = elastix_image_filter.GetTransformParameterMap()
#     transform_file_path = os.path.join(output_dir, transform_file_name)
#     sitk.WriteParameterFile(transform_parameter_map[0], transform_file_path)

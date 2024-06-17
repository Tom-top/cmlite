import os
import shutil
import tempfile

import numpy as np

import SimpleITK as sitk

import utils.utils as ut

def align_images(fixed_image_path, moving_image_path, parameters_path, output_dir, output_file_name,
                 transform_file_name):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Read images
    fixed_image = sitk.ReadImage(fixed_image_path, sitk.sitkFloat32)
    moving_image = sitk.ReadImage(moving_image_path, sitk.sitkFloat32)

    # Setup elastix image filter
    elastix_image_filter = sitk.ElastixImageFilter()
    elastix_image_filter.SetFixedImage(fixed_image)
    elastix_image_filter.SetMovingImage(moving_image)

    # Read parameter map from file
    parameter_map = sitk.ReadParameterFile(parameters_path)
    elastix_image_filter.SetParameterMap(parameter_map)

    # Perform registration
    elastix_image_filter.Execute()

    # Get the result image
    result_image = elastix_image_filter.GetResultImage()

    # Save the result image
    sitk.WriteImage(result_image, os.path.join(output_dir, output_file_name))

    # Get and save the transformation parameters
    transform_parameter_map = elastix_image_filter.GetTransformParameterMap()
    transform_file_path = os.path.join(output_dir, transform_file_name)
    sitk.WriteParameterFile(transform_parameter_map[0], transform_file_path)


def apply_transform(moving_image_path, transform_parameters_paths, output_dir, output_file_name):
    # Read the moving image
    moving_image = sitk.ReadImage(moving_image_path, sitk.sitkFloat32)

    # Read the transformation parameters
    transform_parameter_maps = [sitk.ReadParameterFile(path) for path in transform_parameters_paths]

    # Apply the transformations sequentially
    transformix_image_filter = sitk.TransformixImageFilter()
    transformix_image_filter.SetMovingImage(moving_image)
    for parameter_map in transform_parameter_maps:
        transformix_image_filter.SetTransformParameterMap(parameter_map)
    transformix_image_filter.Execute()

    # Get the result image
    result_image = transformix_image_filter.GetResultImage()

    # Save the result image
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    sitk.WriteImage(result_image, os.path.join(output_dir, output_file_name))


def transform_points(source, sink=None, transform_parameter_files=None, transform_directory=None, indices=False,
                     result_directory=None, temp_file=None, binary=True):
    """Transform coordinates via elastix estimated transformation.

    Arguments
    ---------
    source : str or np.ndarray
        Source of the points.
    sink : str or None
        Sink for transformed points.
    transform_parameter_files : list of str or None
        List of parameter files for the transformations.
        If None, the files are determined from the transform_directory.
    transform_directory : str or None
        Result directory of elastix alignment.
        If None, the transform_parameter_files have to be given.
    indices : bool
        If True, use points as pixel coordinates; otherwise, spatial coordinates.
    result_directory : str or None
        Elastix result directory.
    temp_file : str or None
        Optional file name for the elastix point file.
    binary : bool
        If True, use binary format for point file; otherwise, use text format.

    Returns
    -------
    points : np.ndarray or str
        Array or file name of transformed points.
    """

    def write_points(filename, points, indices=False, binary=True):
        if binary:
            np.savetxt(filename, points, delimiter=' ')
        else:
            np.savetxt(filename, points, delimiter=' ')

    def read_points(filename, binary=True):
        if binary:
            return np.loadtxt(filename, delimiter=' ')
        else:
            return np.loadtxt(filename, delimiter=' ')

    # input point file
    if temp_file is None:
        if binary:
            temp_file = os.path.join(tempfile.gettempdir(), 'elastix_input.bin')
        else:
            temp_file = os.path.join(tempfile.gettempdir(), 'elastix_input.txt')

    delete_point_file = None
    if isinstance(source, str):
        if source[-3:] in ['txt', 'bin']:
            if source[-3:] == 'txt':
                binary = False
            elif source[-3:] == 'bin':
                binary = True
            pointfile = source
        else:
            points = read_points(source, binary=binary)
            pointfile = temp_file
            delete_point_file = temp_file
            write_points(pointfile, points, indices=indices, binary=binary)
    elif isinstance(source, np.ndarray):
        pointfile = temp_file
        delete_point_file = temp_file
        write_points(pointfile, source, indices=indices, binary=binary)
    else:
        raise RuntimeError('transform_points: source not string or array!')

    # result directory
    if result_directory is None:
        outdirname = os.path.join(tempfile.gettempdir(), 'elastix_output')
        delete_result_directory = outdirname
    else:
        outdirname = result_directory
        delete_result_directory = None

    if not os.path.exists(outdirname):
        os.makedirs(outdirname)

    # Ensure transform_parameter_files is a list
    if transform_parameter_files is None:
        raise RuntimeError('transform_points: transform_parameter_files must be provided!')
    if isinstance(transform_parameter_files, str):
        transform_parameter_files = [transform_parameter_files]

    # Sequentially apply transformations
    for i, transform_parameter_file in enumerate(transform_parameter_files):
        print(
            f"Applying transformation {i + 1}/{len(transform_parameter_files)} with parameter file: {transform_parameter_file}")
        transformix = sitk.TransformixImageFilter()

        try:
            parameter_map = sitk.ReadParameterFile(transform_parameter_file)
            transformix.SetTransformParameterMap(parameter_map)
        except Exception as e:
            raise RuntimeError(f"Error reading parameter file {transform_parameter_file}: {e}")

        transformix.SetFixedPointSetFileName(pointfile)
        transformix.SetOutputDirectory(outdirname)

        try:
            transformix.Execute()
        except Exception as e:
            raise RuntimeError(f"Error executing transformix for parameter file {transform_parameter_file}: {e}")

        # Update pointfile to the output of the current transformation
        if binary:
            pointfile = os.path.join(outdirname, 'outputpoints.bin')
        else:
            pointfile = os.path.join(outdirname, 'outputpoints.txt')

    # read data and clean up
    if delete_point_file is not None:
        os.remove(delete_point_file)

    if sink == []:  # return sink as file name
        return pointfile
    else:
        if binary:
            transpoints = read_points(pointfile, binary=True)
        else:
            transpoints = read_points(pointfile, binary=False)

        if delete_result_directory is not None:
            shutil.rmtree(delete_result_directory)

        if sink:
            np.savetxt(sink, transpoints)
            return sink
        else:
            return transpoints


def run_alignments(sample_name, sample_directory, annotation_file, reference_file, **kwargs):
    print("")
    for channel in kwargs["channels_to_segment"]:

        ################################################################################################################
        # 2.1 ALIGN SIGNAL TO AUTO
        ################################################################################################################

        signal_to_auto_directory = os.path.join(sample_directory, f"signal_to_auto_{channel}")
        align_signal_to_auto_affine = dict(fixed_image_path=os.path.join(sample_directory,
                                                                         f"resampled_25um_"
                                                                         f"{kwargs['autofluorescence_channel']}.tif"),
                                           moving_image_path=os.path.join(sample_directory,
                                                                          f"resampled_25um_{channel}.tif"),
                                           parameters_path="resources/alignment/align_affine.txt",
                                           output_dir=signal_to_auto_directory,
                                           output_file_name="affine_result.mhd",
                                           transform_file_name="transform_params_0.txt")
        if not os.path.exists(signal_to_auto_directory):
            ut.print_c(f"[INFO {sample_name}] Running signal to auto (affine) alignment for channel {channel}!")
            align_images(**align_signal_to_auto_affine)
        else:
            ut.print_c(f"[WARNING {sample_name}] Alignment: signal to auto skipped for channel {channel}: "
                       f"signal_to_auto_{channel} folder already exists!")

        ################################################################################################################
        # 2.2 ALIGN AUTO TO TEMPLATE
        ################################################################################################################

        auto_to_template_directory = os.path.join(sample_directory, f"auto_to_template_{channel}")
        align_auto_to_template_affine = dict(fixed_image_path=reference_file,
                                             moving_image_path=os.path.join(sample_directory,
                                                                            f"resampled_25um_"
                                                                            f"{kwargs['autofluorescence_channel']}.tif"),
                                             parameters_path="resources/alignment/align_affine.txt",
                                             output_dir=auto_to_template_directory,
                                             output_file_name="affine_result.mhd",
                                             transform_file_name="transform_params_0.txt")

        align_auto_to_template_bspline = dict(fixed_image_path=reference_file,
                                              moving_image_path=os.path.join(auto_to_template_directory,
                                                                             "affine_result.mhd"),
                                              parameters_path="resources/alignment/align_bspline.txt",
                                              output_dir=auto_to_template_directory,
                                              output_file_name="bspline_result.mhd",
                                              transform_file_name="transform_params_1.txt")

        if not os.path.exists(auto_to_template_directory):
            ut.print_c(f"[INFO {sample_name}] Running auto to template (affine) alignment for channel {channel}!")
            align_images(**align_auto_to_template_affine)
            ut.print_c(f"[INFO {sample_name}] Running auto to template (bspline) alignment for channel {channel}!")
            align_images(**align_auto_to_template_bspline)
        else:
            ut.print_c(f"[WARNING {sample_name}] Alignment: auto to template skipped for channel {channel}: "
                       f"signal_to_auto_{channel} folder already exists!")

        ################################################################################################################
        # 2.3 TRANSFORM SIGNAL TO TEMPLATE
        ################################################################################################################

        signal_to_template_directory = os.path.join(sample_directory, f"signal_to_template_{channel}")
        transform_atlas_parameter = dict(
            moving_image_path=os.path.join(sample_directory, f"resampled_25um_{channel}.tif"),
            output_dir=signal_to_template_directory,
            output_file_name="result.mhd",
            transform_parameters_paths=[os.path.join(auto_to_template_directory,
                                                     f"transform_params_{i}.txt")
                                        for i in range(2)])
        if not os.path.exists(signal_to_template_directory):
            ut.print_c(f"[INFO {sample_name}] Running signal to template transform for channel {channel}!")
            apply_transform(**transform_atlas_parameter)
        else:
            ut.print_c(f"[WARNING {sample_name}] Transforming: signal to template skipped for channel {channel}: "
                       f"signal_to_template_{channel} folder already exists!")

        ################################################################################################################
        # 2.4 ALIGN TEMPLATE TO AUTO
        ################################################################################################################

        template_to_auto_directory = os.path.join(sample_directory, f"template_to_auto__{channel}")
        align_template_to_auto_affine = dict(fixed_image_path=os.path.join(sample_directory,
                                                                           f"resampled_25um_"
                                                                           f"{kwargs['autofluorescence_channel']}.tif"),
                                             moving_image_path=reference_file,
                                             parameters_path="resources/alignment/align_affine.txt",
                                             output_dir=template_to_auto_directory,
                                             output_file_name="affine_result.mhd",
                                             transform_file_name="transform_params_0.txt")

        align_template_to_auto_bspline = dict(fixed_image_path=os.path.join(sample_directory,
                                                                            f"resampled_25um_"
                                                                            f"{kwargs['autofluorescence_channel']}.tif"),
                                              moving_image_path=os.path.join(template_to_auto_directory,
                                                                             "affine_result.mhd"),
                                              parameters_path="resources/alignment/align_bspline.txt",
                                              output_dir=template_to_auto_directory,
                                              output_file_name="bspline_result.mhd",
                                              transform_file_name="transform_params_1.txt")

        if not os.path.exists(template_to_auto_directory):
            ut.print_c(f"[INFO {sample_name}] Running template to auto (affine) alignment for channel {channel}!")
            align_images(**align_template_to_auto_affine)
            ut.print_c(f"[INFO {sample_name}] Running template to auto (bspline) alignment for channel {channel}!")
            align_images(**align_template_to_auto_bspline)
        else:
            ut.print_c(f"[WARNING {sample_name}] Alignment: template to auto skipped for channel {channel}: "
                       f"signal_to_auto_{channel} folder already exists!")

        ################################################################################################################
        # 2.5 TRANSFORM ATLAS TO AUTO
        ################################################################################################################

        # Toggling pixel interpolation off during transform
        atlas_to_auto_directory = os.path.join(sample_directory, f"atlas_to_auto_{channel}")
        with open(os.path.join(template_to_auto_directory, "transform_params_1.txt"), 'r') \
                as file:
            data = file.read()
            data = data.replace("(FinalBSplineInterpolationOrder 3)", "(FinalBSplineInterpolationOrder 0)")
        with open(os.path.join(template_to_auto_directory, "transform_params_1_nointer.txt"),
                  'w') as file:
            file.write(data)

        transform_atlas_parameter = dict(
            moving_image_path=annotation_file,
            output_dir=atlas_to_auto_directory,
            output_file_name="result.mhd",
            transform_parameters_paths=[os.path.join(template_to_auto_directory, f"transform_params_0.txt"),
                                        os.path.join(template_to_auto_directory, f"transform_params_1_nointer.txt")])

        if not os.path.exists(atlas_to_auto_directory):
            ut.print_c(f"[INFO {sample_name}] Running atlas to auto transform for channel {channel}!")
            apply_transform(**transform_atlas_parameter)
        else:
            ut.print_c(f"[WARNING {sample_name}] Transforming: atlas to auto skipped for channel {channel}: "
                       f"signal_to_template_{channel} folder already exists!")

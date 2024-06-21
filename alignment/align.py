import os
import sys
import tempfile
import subprocess
import platform
import multiprocessing as mp

import numpy as np
from io import UnsupportedOperation

import SimpleITK as sitk

import utils.utils as ut
import utils.exceptions as excep

elastix_lib_path = "/home/imaging/PycharmProjects/cmlite/external/elastix/build/bin"  # Change this to the directory containing libANNlib-4.9.so.1
elastix_binary = "/home/imaging/PycharmProjects/cmlite/external/elastix/build/bin/elastix"
transformix_binary = "/home/imaging/PycharmProjects/cmlite/external/elastix/build/bin/transformix"

tempdir = tempfile.gettempdir()
elastix_output_folder = os.path.join(tempdir, "elastix_output")
if not os.path.exists(elastix_output_folder):
    os.mkdir(elastix_output_folder)


def initialize_elastix():
    global elastix_lib_path
    os_name = platform.system().lower()
    if os_name.startswith('linux'):
        lib_var_name = 'LD_LIBRARY_PATH'
    elif os_name.startswith('darwin'):
        lib_var_name = 'DYLD_LIBRARY_PATH'
    else:
        raise ValueError(f'Unknown or unsupported OS {os_name}')

    ut.print_c(f'[INFO] OS: {os_name}, library variable name: {lib_var_name}')

    if lib_var_name in os.environ:
        lib_path = os.environ[lib_var_name]
        ut.print_c(f'[INFO] Variable {lib_var_name} exists, patching with {lib_path}')
        if elastix_lib_path not in lib_path.split(':'):
            os.environ[lib_var_name] = f'{elastix_lib_path}:{lib_path}'
    else:
        ut.print_c(f'[INFO] Variable {lib_var_name} not found, adding elastix lib folder')
        os.environ[lib_var_name] = elastix_lib_path


initialize_elastix()


def align_images(fixed_image_path, moving_image_path, affine_parameter_file, bspline_parameter_file=None,
                 output_dir=None, processes=None,
                 workspace=None, moving_landmarks_path=None, fixed_landmarks_path=None):
    """
    Align images using elastix, estimates a transformation :math:`T:` fixed image :math:`\\rightarrow` moving image.

    Arguments
    ---------
    fixed_image_path : str
      Image source of the fixed image (typically the reference image).
    moving_image_path : str
      Image source of the moving image (typically the image to be registered).
    affine_parameter_file : str or None
      Elastix parameter file for the primary affine transformation.
    bspline_parameter_file : str or None
      Elastix parameter file for the secondary non-linear transformation.
    output_dir : str or None
      Elastic result directory.
    processes : int or None
      Number of threads to use.

    Returns
    -------
    result_directory : str
      Path to elastix result directory.
    """

    processes = processes if processes is not None else mp.cpu_count()

    # result directory
    output_dir = output_dir if output_dir is not None else tempfile.gettempdir()

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # run elastix
    cmd = [elastix_binary, '-threads', str(processes), '-m', f'{moving_image_path}', '-f',
           f'{fixed_image_path}']  # We quote all the paths for spaces
    if affine_parameter_file is not None:
        cmd.extend(['-p', f'{affine_parameter_file}'])
    if bspline_parameter_file is not None:
        cmd.extend(['-p', f'{bspline_parameter_file}'])
    if moving_landmarks_path is not None or fixed_landmarks_path is not None:
        cmd.extend(['-mp', f'{moving_landmarks_path}', '-fp', f'{fixed_landmarks_path}'])
    cmd.extend(['-out', f'{output_dir}'])

    try:
        with subprocess.Popen(cmd, stdout=sys.stdout,
                              stderr=sys.stdout) as proc:  # FIXME: check if we need an "if not sys.stdout.fileno"
            if workspace is not None:
                workspace.process = proc
    except UnsupportedOperation:
        try:
            subprocess.Popen(cmd)
        except (subprocess.SubprocessError, OSError) as err:
            raise excep.ClearMapException(f'Align: failed executing: {" ".join(cmd)}') from err
    except (subprocess.SubprocessError, OSError) as err:
        raise excep.ClearMapException(f'Align: failed executing: {" ".join(cmd)}') from err
    finally:
        if workspace is not None:
            workspace.process = None

    return output_dir


def apply_transform(moving_image_path, transform_parameters_paths, output_dir, output_file_name):
    # Read the moving image
    moving_image = sitk.ReadImage(moving_image_path, sitk.sitkFloat32)

    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Apply each transformation sequentially
    for i, parameter_path in enumerate(transform_parameters_paths):
        # Read the transformation parameters
        parameter_map = sitk.ReadParameterFile(parameter_path)

        # Set up the TransformixImageFilter with the current moving image and parameter map
        transformix_image_filter = sitk.TransformixImageFilter()
        transformix_image_filter.SetMovingImage(moving_image)
        transformix_image_filter.SetTransformParameterMap(parameter_map)

        # Apply the transformation
        transformix_image_filter.Execute()

        # Get the result image, which will be the input for the next transformation
        moving_image = transformix_image_filter.GetResultImage()

        # Optionally, save the intermediate result image (useful for debugging)
        intermediate_output_file_name = f"intermediate_result_{i}.nii"
        sitk.WriteImage(moving_image, os.path.join(output_dir, intermediate_output_file_name))

    # Save the final result image
    sitk.WriteImage(moving_image, os.path.join(output_dir, output_file_name))


def write_points(filename, points, indices=False, binary=True):
    """Write points as elastix/transformix point file."""
    if binary:
        with open(filename, 'wb') as pointfile:
            if indices:
                np.array(1, dtype=np.int64).tofile(pointfile)
            else:
                np.array(0, dtype=np.int64).tofile(pointfile)

            num_points = np.array(len(points), dtype=np.int64)
            num_points.tofile(pointfile)

            points = np.asarray(points, dtype=np.double)
            points.tofile(pointfile)
    else:
        with open(filename, 'w') as pointfile:
            if indices:
                pointfile.write('index\n')
            else:
                pointfile.write('point\n')

            pointfile.write(str(points.shape[0]) + '\n')
            np.savetxt(pointfile, points, delimiter=' ', newline='\n', fmt='%.5e')

    return filename


def read_points(filename, indices=False, binary=True):
    """Parses the output points from the output file of transformix."""
    if binary:
        with open(filename, 'rb') as f:
            index = np.fromfile(f, dtype=np.int64, count=1)[0]
            if index == 0:
                indices = False
            else:
                indices = True

            num_points = np.fromfile(f, dtype=np.int64, count=1)[0]
            if num_points == 0:
                return np.zeros((0, 3))

            points = np.fromfile(f, dtype=np.double)
            points = np.reshape(points, (num_points, 3))

        return points
    else:  # text file
        with open(filename) as f:
            lines = f.readlines()

        num_points = len(lines)

        if num_points == 0:
            return np.zeros((0, 3))

        points = np.zeros((num_points, 3))
        k = 0
        for line in lines:
            ls = line.split()
            if indices:
                for i in range(0, 3):
                    points[k, i] = float(ls[i + 22])
            else:
                for i in range(0, 3):
                    points[k, i] = float(ls[i + 30])

            k += 1

        return points


def transform_points_with_transformix(in_point_file, out_dir, transform_parameter_file, transformix_input=False):
    global transformix_binary
    # Set the LD_LIBRARY_PATH to include the directory with the required libraries

    transformed_points_file = os.path.join(out_dir, "outputpoints.txt")
    if transformix_input:
        read_transformix_output_file(in_point_file, transformed_points_file)

    # Create the command to call Transformix
    cmd = f'{transformix_binary} -def {in_point_file} -out {out_dir} -tp {transform_parameter_file}'
    print(cmd)

    # Run the command
    res = os.system(cmd)

    if res != 0:
        raise RuntimeError('transform_data: failed executing: ' + cmd)

    # Read the transformed points from the output directory
    transformed_points = read_points(transformed_points_file, indices=False, binary=False)

    return transformed_points, transformed_points_file


def read_transformix_output_file(input_file, output_file):
    with open(input_file, 'r') as file:
        lines = file.readlines()

    total_points = len(lines)
    output_lines = [f"point\n{total_points}"]

    for line in lines:
        parts = line.split(';')
        input_point_str = parts[1].split('=')[1].strip()
        input_point_values = input_point_str.strip('[]').split()
        formatted_point = ' '.join([f"{float(value):.5e}" for value in input_point_values])
        output_lines.append(formatted_point)

    with open(output_file, 'w') as file:
        file.write('\n'.join(output_lines))


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
                                           affine_parameter_file="resources/alignment/align_affine.txt",
                                           bspline_parameter_file=None,
                                           output_dir=signal_to_auto_directory,
                                           )

        if not os.path.exists(signal_to_auto_directory):
            ut.print_c(f"[INFO {sample_name}] Running signal to auto (affine) alignment for channel {channel}!")
            align_images(**align_signal_to_auto_affine)
        else:
            ut.print_c(f"[WARNING {sample_name}] Alignment: signal to auto skipped for channel {channel}: "
                       f"signal_to_auto_{channel} folder already exists!")

        ################################################################################################################
        # 2.1 ALIGN AUTO TO SIGNAL
        ################################################################################################################

        auto_to_signal_directory = os.path.join(sample_directory, f"auto_to_signal_{channel}")
        align_auto_to_signal_affine = dict(fixed_image_path=os.path.join(sample_directory,
                                                                         f"resampled_25um_{channel}.tif"),
                                           moving_image_path=os.path.join(sample_directory,
                                                                          f"resampled_25um_"
                                                                          f"{kwargs['autofluorescence_channel']}.tif"),
                                           affine_parameter_file="resources/alignment/align_affine.txt",
                                           bspline_parameter_file=None,
                                           output_dir=auto_to_signal_directory,
                                           )
        if not os.path.exists(auto_to_signal_directory):
            ut.print_c(f"[INFO {sample_name}] Running auto to signal (affine) alignment for channel {channel}!")
            align_images(**align_auto_to_signal_affine)
        else:
            ut.print_c(f"[WARNING {sample_name}] Alignment: auto to signal skipped for channel {channel}: "
                       f"auto to signal_{channel} folder already exists!")

        ################################################################################################################
        # 2.2 ALIGN AUTO TO REFERENCE
        ################################################################################################################

        auto_to_reference_directory = os.path.join(sample_directory, f"auto_to_reference_{channel}")
        align_auto_to_reference = dict(fixed_image_path=reference_file,
                                       moving_image_path=os.path.join(sample_directory,
                                                                      f"resampled_25um_"
                                                                      f"{kwargs['autofluorescence_channel']}.tif"),
                                       affine_parameter_file="resources/alignment/align_affine.txt",
                                       bspline_parameter_file="resources/alignment/align_bspline.txt",
                                       output_dir=auto_to_reference_directory,
                                       )

        if not os.path.exists(auto_to_reference_directory):
            ut.print_c(f"[INFO {sample_name}] Running auto to reference alignment for channel {channel}!")
            align_images(**align_auto_to_reference)
        else:
            ut.print_c(f"[WARNING {sample_name}] Alignment: auto to reference skipped for channel {channel}: "
                       f"signal_to_auto_{channel} folder already exists!")

        ################################################################################################################
        # 2.4 ALIGN REFERENCE TO AUTO
        ################################################################################################################

        reference_to_auto_directory = os.path.join(sample_directory, f"reference_to_auto_{channel}")
        align_reference_to_auto = dict(fixed_image_path=os.path.join(sample_directory,
                                                                     f"resampled_25um_"
                                                                     f"{kwargs['autofluorescence_channel']}.tif"),
                                       moving_image_path=reference_file,
                                       affine_parameter_file="resources/alignment/align_affine.txt",
                                       bspline_parameter_file="resources/alignment/align_bspline.txt",
                                       output_dir=reference_to_auto_directory,
                                       )

        if not os.path.exists(reference_to_auto_directory):
            ut.print_c(f"[INFO {sample_name}] Running reference to auto alignment for channel {channel}!")
            align_images(**align_reference_to_auto)
        else:
            ut.print_c(f"[WARNING {sample_name}] Alignment: reference to auto skipped for channel {channel}: "
                       f"signal_to_auto_{channel} folder already exists!")

        ################################################################################################################
        # 2.3 TRANSFORM SIGNAL TO REFERENCE
        ################################################################################################################

        signal_to_reference_directory = os.path.join(sample_directory, f"signal_to_template_{channel}")
        transform_atlas_parameter = dict(
            moving_image_path=os.path.join(sample_directory, f"resampled_25um_{channel}.tif"),
            output_dir=signal_to_reference_directory,
            output_file_name="result_1.mhd",
            transform_parameters_paths=[os.path.join(auto_to_reference_directory,
                                                     f"transform_params_{i}.txt")
                                        for i in range(2)])
        if not os.path.exists(signal_to_reference_directory):
            ut.print_c(f"[INFO {sample_name}] Running signal to reference transform for channel {channel}!")
            apply_transform(**transform_atlas_parameter)
        else:
            ut.print_c(f"[WARNING {sample_name}] Transforming: signal to reference skipped for channel {channel}: "
                       f"signal_to_template_{channel} folder already exists!")

        ################################################################################################################
        # 2.5 TRANSFORM ATLAS TO AUTO
        ################################################################################################################

        # Toggling pixel interpolation off during transform
        atlas_to_auto_directory = os.path.join(sample_directory, f"atlas_to_auto_{channel}")

        with open(os.path.join(reference_to_auto_directory, "TransformParameters.0.txt"), 'r') \
                as file:
            data = file.read()
            data = data.replace("(FinalBSplineInterpolationOrder 3.000000)",
                                "(FinalBSplineInterpolationOrder 0.000000)")
        with open(os.path.join(reference_to_auto_directory, "TransformParameters_ni.0.txt"),
                  'w') as file:
            file.write(data)

        with open(os.path.join(reference_to_auto_directory, "TransformParameters.1.txt"), 'r') \
                as file:
            data = file.read()
            data = data.replace("(FinalBSplineInterpolationOrder 3.000000)",
                                "(FinalBSplineInterpolationOrder 0.000000)")
        with open(os.path.join(reference_to_auto_directory, "TransformParameters_ni.1.txt"),
                  'w') as file:
            file.write(data)

        transform_atlas_parameter = dict(
            moving_image_path=annotation_file,
            output_dir=atlas_to_auto_directory,
            output_file_name="result_1.mhd",
            transform_parameters_paths=[os.path.join(reference_to_auto_directory, f"TransformParameters_ni.0.txt"),
                                        os.path.join(reference_to_auto_directory, f"TransformParameters_ni.1.txt")])

        if not os.path.exists(atlas_to_auto_directory):
            ut.print_c(f"[INFO {sample_name}] Running atlas to auto transform for channel {channel}!")
            apply_transform(**transform_atlas_parameter)
        else:
            ut.print_c(f"[WARNING {sample_name}] Transforming: atlas to auto skipped for channel {channel}: "
                       f"atlas_to_auto_{channel} folder already exists!")

import os
import re
import sys
import json
import tempfile
import shutil
import subprocess
import platform
import multiprocessing as mp

import numpy as np
from io import UnsupportedOperation
import ccf_streamlines.projection as ccfproj
import skimage.io as skio
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Qt5Agg")  # Headless mode

import settings
import utils.utils as ut
import utils.exceptions as excep

import IO.IO as io

cmlite_env_path = os.getcwd()
if settings.platform_name == "linux":
    elastix_lib_path = os.path.join(cmlite_env_path,
                                    f"external/elastix/{settings.platform_name}/build/bin")  # Change this to the directory containing libANNlib-4.9.so.1
    elastix_binary = os.path.join(cmlite_env_path, f"external/elastix/{settings.platform_name}/build/bin/elastix")
    transformix_binary = os.path.join(cmlite_env_path,
                                      f"external/elastix/{settings.platform_name}/build/bin/transformix")
elif settings.platform_name == "windows":
    elastix_lib_path = os.path.join(cmlite_env_path,
                                    f"external/elastix/{settings.platform_name}")  # Change this to the directory containing libANNlib-4.9.so.1
    elastix_binary = os.path.join(cmlite_env_path, f"external/elastix/{settings.platform_name}/elastix.exe")
    transformix_binary = os.path.join(cmlite_env_path, f"external/elastix/{settings.platform_name}/transformix.exe")

tempdir = tempfile.gettempdir()
elastix_output_folder = os.path.join(tempdir, "elastix_output")
if not os.path.exists(elastix_output_folder):
    os.mkdir(elastix_output_folder)


def initialize_elastix():
    global elastix_lib_path
    os_name = platform.system().lower()
    if os_name.startswith('linux') or os_name.startswith('windows'):
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


def align_images(fixed_image_path, moving_image_path, affine_parameter_file=None, bspline_parameter_file=None,
                 output_dir=None, processes=None, workspace=None, moving_landmarks_path=None,
                 fixed_landmarks_path=None):
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

    if settings.platform_name == "windows":
        cmd = [i.replace('/', '\\') for i in cmd]

    print(" ".join(cmd))

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


def transform_file(result_directory):
    """Finds and returns the transformation parameter file.

    Arguments
    ---------
    result_directory : str
      Path to directory of elastix results.

    Returns
    -------
    filename : str
      File name of the first transformation parameter file.

    Notes
    -----
    In case of multiple transformation parameter files the top level file is
    returned.
    """

    files = os.listdir(result_directory)
    files = [x for x in files if re.match('TransformParameters.\d.txt', x)]
    files.sort()

    if not files:
        raise RuntimeError('Cannot find a valid transformation file in %r!' % result_directory)

    return os.path.join(result_directory, files[-1])


def transform_directory_and_file(transform_parameter_file=None, transform_directory=None):
    """Determines transformation directory and file from either.

    Arguments
    ---------
    transform_parameter_file : str or None
      File name of the transformation parameter file.
    transform_directory : str or None
      Directory to the transformation parameter.

    Returns
    -------
    transform_parameter_file : str
      File name of the transformation parameter file.
    transform_directory : str
      Directory to the transformation parameter.

    Notes
    -----
    Only one of the two arguments need to be specified.
    """

    if not transform_parameter_file:
        if not transform_directory:
            raise ValueError('Neither the alignment directory nor the transformation parameter file is specified!')
        transform_parameter_dir = transform_directory
        transform_parameter_file = transform_file(transform_parameter_dir)
    else:
        transform_parameter_dir = os.path.split(transform_parameter_file)
        transform_parameter_dir = transform_parameter_dir[0]
        transform_parameter_file = transform_parameter_file

    return transform_parameter_dir, transform_parameter_file


def set_path_transform_files(result_directory):
    """Replaces relative with absolute path in the parameter files in the result directory.

    Arguments
    ---------
    result_directory : str
      Path to directory of elastix results.

    Notes
    -----
    When elastix is not run in the directory of the transformation files
    the aboslute path needs to be given in each transformation file
    to point to the subsequent transformation files. This is done via this
    routine.
    """

    files = os.listdir(result_directory)
    files = [x for x in files if re.match('TransformParameters.\d.txt', x)]
    files.sort()

    if not files:
        raise RuntimeError('Cannot find a valid transformation file in %r!' % result_directory)

    rec = re.compile("\(InitialTransformParametersFileName \"(?P<parname>.*)\"\)")

    for f in files:
        fh, tmpfn = tempfile.mkstemp()
        ff = os.path.join(result_directory, f)

        with open(tmpfn, 'w') as newfile, open(ff) as parfile:
            for line in parfile:
                m = rec.match(line)
                if m != None:
                    pn = m.group('parname')
                    if pn != 'NoInitialTransform':
                        pathn, filen = os.path.split(pn)
                        filen = os.path.join(result_directory, filen)
                        newfile.write(line.replace(pn, filen))
                    else:
                        newfile.write(line)
                else:
                    newfile.write(line)

        os.close(fh)
        os.remove(ff)
        shutil.move(tmpfn, ff)


def result_data_file(result_directory):
    """Returns the mhd result file in a result directory.

    Arguments
    ---------
    result_directory : str
      Path to elastix result directory.

    Returns
    -------
    result_file : str
      The mhd file in the result directory.
    """
    files = os.listdir(result_directory)
    files = [x for x in files if re.match('.*.mhd', x)]
    files.sort()

    if not files:
        raise RuntimeError('Cannot find a valid result data file in ' + result_directory)

    return os.path.join(result_directory, files[0])


def transform_images(source, sink=[], transform_parameter_file=None, transform_directory=None, result_directory=None):
    """Transform a raw data set to reference using the elastix alignment results.

    Arguments
    ---------
    source : str or array
      Image source to be transformed.
    sink : str, [] or None
      Image sink to save transformed image to. If [] return the default name
      of the data file generated by transformix.
    transform_parameter_file : str or None
      Parameter file for the primary transformation.
      If None, the file is determined from the transform_directory.
    transform_directory : str or None
      Result directory of elastix alignment.
      If None the transform_parameter_file has to be given.
    result_directory : str or None
      The directorty for the transformix results.

    Returns
    -------
    transformed : array or st
      Array or file name of the transformed data.

    Note
    ----
    If the map determined by elastix is
    :math:`T: \\mathrm{fixed} \\rightarrow \\mathrm{moving}`,
    transformix on data works as :math:`T^{-1}(\\mathrm{data})`.
    """

    # image
    source = io.as_source(source)
    if isinstance(source, io.tif.Source):
        imgname = source.location
        delete_image = None
    else:
        imgname = os.path.join(tempfile.gettempdir(), 'elastix_input.tif')
        io.write(source, imgname)
        delete_image = imgname

    # result directory
    delete_result_directory = None
    if result_directory == None:
        resultdirname = os.path.join(tempfile.gettempdir(), 'elastix_output')
        delete_result_directory = resultdirname
    else:
        resultdirname = result_directory

    if not os.path.exists(resultdirname):
        os.makedirs(resultdirname)

    # tranformation parameter
    transform_parameter_dir, transform_parameter_file = transform_directory_and_file(
        transform_parameter_file=transform_parameter_file, transform_directory=transform_directory)

    if not transform_parameter_file:  # FIXME: unstable
        set_path_transform_files(transform_parameter_dir)

    # transformix -in inputImage.ext -out outputDirectory -tp TransformParameters.txx
    cmd = '%s -in %s -out %s -tp %s' % (transformix_binary, imgname, resultdirname, transform_parameter_file)

    res = os.system(cmd)

    if res != 0:
        raise RuntimeError('transform_data: failed executing: ' + cmd)

    # read data and clean up
    if delete_image is not None:
        os.remove(delete_image)

    if sink == []:
        return result_data_file(resultdirname)
    elif sink is None:
        resultfile = result_data_file(resultdirname)
        result = io.read(resultfile)
    elif isinstance(sink, str):
        resultfile = result_data_file(resultdirname)
        result = io.convert(resultfile, sink)
    else:
        raise RuntimeError('transform_data: sink not valid!')

    if delete_result_directory is not None:
        shutil.rmtree(delete_result_directory)

    return result


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


def run_alignments(sample_name, sample_directory, annotation_files, reference_files, **kwargs):
    print("")
    for channel in kwargs["study_params"]["channels_to_segment"]:

        ################################################################################################################
        # 2.1 ALIGN SIGNAL TO AUTO
        ################################################################################################################

        signal_to_auto_directory = os.path.join(sample_directory, f"signal_to_auto_{channel}")
        align_signal_to_auto_affine = dict(fixed_image_path=os.path.join(sample_directory,
                                                                         f"resampled_25um_"
                                                                         f"{kwargs['study_params']['autofluorescence_channel']}.tif"),
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
                                                                          f"{kwargs['study_params']['autofluorescence_channel']}.tif"),
                                           affine_parameter_file="resources/alignment/align_affine.txt",
                                           bspline_parameter_file=None,
                                           output_dir=auto_to_signal_directory,
                                           )
        if not os.path.exists(auto_to_signal_directory):
            ut.print_c(f"[INFO {sample_name}] Running auto to signal (affine) alignment for channel {channel}!")
            align_images(**align_auto_to_signal_affine)
            # generate_alignment_overlay(os.path.join(sample_directory,
            #                                         f"resampled_25um_{channel}.tif"),
            #                            os.path.join(auto_to_signal_directory, "result.0.mhd"),
            #                            os.path.join(auto_to_signal_directory, "auto_to_signal_affine.tif"))
        else:
            ut.print_c(f"[WARNING {sample_name}] Alignment: auto to signal skipped for channel {channel}: "
                       f"auto to signal_{channel} folder already exists!")

        for atlas, annotation, reference in zip(kwargs["study_params"]["atlas_to_use"],
                                                annotation_files,
                                                reference_files):

            ################################################################################################################
            # 2.2 ALIGN AUTO TO REFERENCE
            ################################################################################################################

            auto_to_reference_directory = os.path.join(sample_directory, f"{atlas}_auto_to_reference_{channel}")
            align_auto_to_reference = dict(fixed_image_path=reference,
                                           moving_image_path=os.path.join(sample_directory,
                                                                          f"resampled_25um_"
                                                                          f"{kwargs['study_params']['autofluorescence_channel']}.tif"),
                                           affine_parameter_file="resources/alignment/align_affine.txt",
                                           bspline_parameter_file="resources/alignment/align_bspline.txt",
                                           output_dir=auto_to_reference_directory,
                                           )

            if not os.path.exists(auto_to_reference_directory):
                ut.print_c(f"[INFO {sample_name}] Running auto to {atlas} reference alignment for channel {channel}!")
                align_images(**align_auto_to_reference)
            else:
                ut.print_c(
                    f"[WARNING {sample_name}] Alignment: auto to {atlas} reference skipped for channel {channel}: "
                    f"signal_to_auto_{channel} folder already exists!")

            ################################################################################################################
            # 2.4 ALIGN REFERENCE TO AUTO
            ################################################################################################################

            reference_to_auto_directory = os.path.join(sample_directory, f"{atlas}_reference_to_auto_{channel}")
            align_reference_to_auto = dict(fixed_image_path=os.path.join(sample_directory,
                                                                         f"resampled_25um_"
                                                                         f"{kwargs['study_params']['autofluorescence_channel']}.tif"),
                                           moving_image_path=reference,
                                           affine_parameter_file="resources/alignment/align_affine.txt",
                                           bspline_parameter_file="resources/alignment/align_bspline.txt",
                                           output_dir=reference_to_auto_directory,
                                           )

            if not os.path.exists(reference_to_auto_directory):
                ut.print_c(f"[INFO {sample_name}] Running {atlas} reference to auto alignment for channel {channel}!")
                align_images(**align_reference_to_auto)
            else:
                ut.print_c(
                    f"[WARNING {sample_name}] Alignment: {atlas} reference to auto skipped for channel {channel}: "
                    f"signal_to_auto_{channel} folder already exists!")

            ################################################################################################################
            # 2.3 TRANSFORM SIGNAL TO REFERENCE
            ################################################################################################################

            signal_to_reference_directory = os.path.join(sample_directory, f"{atlas}_signal_to_reference_{channel}")
            transform_atlas_parameter = dict(
                source=os.path.join(sample_directory, f"resampled_25um_{channel}.tif"),
                result_directory=signal_to_reference_directory,
                transform_parameter_file=os.path.join(auto_to_reference_directory, f"TransformParameters.1.txt"))
            if not os.path.exists(signal_to_reference_directory):
                ut.print_c(f"[INFO {sample_name}] Running signal to {atlas} reference transform for channel {channel}!")
                transform_images(**transform_atlas_parameter)
            else:
                ut.print_c(
                    f"[WARNING {sample_name}] Transforming: signal to {atlas} reference skipped for channel {channel}: "
                    f"signal_to_template_{channel} folder already exists!")

            ################################################################################################################
            # 2.5 TRANSFORM ATLAS TO AUTO
            ################################################################################################################

            # Toggling pixel interpolation off during transform
            atlas_to_auto_directory = os.path.join(sample_directory, f"{atlas}_atlas_to_auto_{channel}")

            with open(os.path.join(reference_to_auto_directory, "TransformParameters.0.txt"), 'r') \
                    as file:
                data = file.read()
                data = data.replace("(FinalBSplineInterpolationOrder 3)",
                                    "(FinalBSplineInterpolationOrder 0)")
            with open(os.path.join(reference_to_auto_directory, "TransformParameters_ni.0.txt"),
                      'w') as file:
                file.write(data)

            with open(os.path.join(reference_to_auto_directory, "TransformParameters.1.txt"), 'r') \
                    as file:
                data = file.read()
                data = data.replace("(FinalBSplineInterpolationOrder 3)",
                                    "(FinalBSplineInterpolationOrder 0)")
            with open(os.path.join(reference_to_auto_directory, "TransformParameters_ni.1.txt"),
                      'w') as file:
                file.write(data)

            transform_atlas_parameter = dict(
                source=annotation,
                result_directory=atlas_to_auto_directory,
                transform_parameter_file=os.path.join(reference_to_auto_directory, f"TransformParameters_ni.1.txt"))
            if not os.path.exists(atlas_to_auto_directory):
                ut.print_c(f"[INFO {sample_name}] Running {atlas} atlas to auto transform for channel {channel}!")
                transform_images(**transform_atlas_parameter)
            else:
                ut.print_c(
                    f"[WARNING {sample_name}] Transforming: {atlas} atlas to auto skipped for channel {channel}: "
                    f"atlas_to_auto_{channel} folder already exists!")

            ############################################################################################################
            # 2.6 [OPTIONAL] TRANSFORM SIGNAL TO REFERENCE (ABA) AT 10um: FLATMAP GENERATION
            ############################################################################################################

            if atlas in ["mouse_gubra", "mouse_aba"]:

                ########################################################################################################
                # 2.6.1 ALIGN AUTO TO REFERENCE (ABA)
                ########################################################################################################

                reference_split = reference.split(".")
                reference_split_u = reference_split[0].split("_")
                reference_10um_path = ("_".join(reference_split_u[:-4]) + "_10um_" + "_".join(reference_split_u[-4:])
                                       + "." + reference_split[-1])
                auto_to_reference_aba_directory = os.path.join(sample_directory,
                                                               f"{atlas}_auto_to_reference_10um_{channel}")
                align_auto_to_reference_10um = dict(fixed_image_path=reference_10um_path,
                                                    moving_image_path=os.path.join(sample_directory,
                                                                                   f"resampled_10um_"
                                                                                   f"{kwargs['study_params']['autofluorescence_channel']}.tif"),
                                                    affine_parameter_file="resources/alignment/align_affine_10um.txt",
                                                    bspline_parameter_file="resources/alignment/align_bspline.txt",
                                                    output_dir=auto_to_reference_aba_directory,
                                                    )

                if not os.path.exists(auto_to_reference_aba_directory):
                    ut.print_c(
                        f"[INFO {sample_name}] Running auto to {atlas} reference (10um) alignment for channel {channel}!")
                    align_images(**align_auto_to_reference_10um)
                else:
                    ut.print_c(
                        f"[WARNING {sample_name}] Alignment: auto to {atlas} reference (10um) skipped for channel {channel}: "
                        f"auto_to_reference_10um_{channel} folder already exists!")

                ########################################################################################################
                # 2.6.2 TRANSFORM SIGNAL TO REFERENCE
                ########################################################################################################

                signal_to_reference_10um_directory = os.path.join(sample_directory,
                                                                  f"{atlas}_signal_to_reference_10um_{channel}")
                transform_atlas_parameter = dict(
                    source=os.path.join(sample_directory, f"resampled_10um_{channel}.tif"),
                    result_directory=signal_to_reference_10um_directory,
                    transform_parameter_file=os.path.join(auto_to_reference_aba_directory,
                                                          f"TransformParameters.1.txt"))
                if not os.path.exists(signal_to_reference_10um_directory):
                    ut.print_c(
                        f"[INFO {sample_name}] Running signal to {atlas} reference (10um) transform for channel {channel}!")
                    transform_images(**transform_atlas_parameter)
                else:
                    ut.print_c(
                        f"[WARNING {sample_name}] Transforming: signal to {atlas} reference (10um) skipped for channel {channel}: "
                        f"signal_to_reference_10um_{channel} folder already exists!")

                if atlas == "mouse_aba":

                    ####################################################################################################
                    # 2.6.3 GENERATE FLATMAPS
                    ####################################################################################################

                    flatmap_figures = [os.path.join(signal_to_reference_10um_directory,
                                                    "cortical_flatmap_all_layers.png"),
                                       os.path.join(signal_to_reference_10um_directory,
                                                    "cortical_flatmap_all_layers_0_3000.png")]

                    if not all([os.path.exists(i) for i in flatmap_figures]):
                        cortical_flatmaps_directory = r"resources\cortical_flatmaps"

                        proj_bf = ccfproj.Isocortex2dProjector(
                            os.path.join(cortical_flatmaps_directory, "flatmap_butterfly.h5"),
                            os.path.join(cortical_flatmaps_directory, "surface_paths_10_v3.h5"),
                            hemisphere="both",
                            view_space_for_other_hemisphere='flatmap_butterfly',
                        )

                        bf_boundary_finder = ccfproj.BoundaryFinder(
                            projected_atlas_file=os.path.join(cortical_flatmaps_directory, "flatmap_butterfly.nrrd"),
                            labels_file=os.path.join(cortical_flatmaps_directory, "labelDescription_ITKSNAPColor.txt"),
                        )

                        bf_left_boundaries = bf_boundary_finder.region_boundaries()

                        bf_right_boundaries = bf_boundary_finder.region_boundaries(
                            hemisphere='right_for_both',
                            view_space_for_other_hemisphere='flatmap_butterfly',
                        )

                        with open(os.path.join(cortical_flatmaps_directory, "avg_layer_depths.json"), "r") as f:
                            layer_tops = json.load(f)

                        layer_thicknesses = {
                            'Isocortex layer 1': layer_tops['2/3'],
                            'Isocortex layer 2/3': layer_tops['4'] - layer_tops['2/3'],
                            'Isocortex layer 4': layer_tops['5'] - layer_tops['4'],
                            'Isocortex layer 5': layer_tops['6a'] - layer_tops['5'],
                            'Isocortex layer 6a': layer_tops['6b'] - layer_tops['6a'],
                            'Isocortex layer 6b': layer_tops['wm'] - layer_tops['6b'],
                        }

                        proj_butterfly_slab = ccfproj.Isocortex3dProjector(
                            os.path.join(cortical_flatmaps_directory, "flatmap_butterfly.h5"),
                            os.path.join(cortical_flatmaps_directory, "surface_paths_10_v3.h5"),
                            hemisphere="both",
                            view_space_for_other_hemisphere='flatmap_butterfly',
                            thickness_type="normalized_layers",  # each layer will have the same thickness everwhere
                            layer_thicknesses=layer_thicknesses,
                            streamline_layer_thickness_file=os.path.join(cortical_flatmaps_directory,
                                                                         "cortical_layers_10_v2.h5"),
                        )

                        auto_to_ABA_10um_path = os.path.join(signal_to_reference_10um_directory, "result.mhd")
                        auto_to_ABA_10um = skio.imread(auto_to_ABA_10um_path, plugin='simpleitk')
                        min_val = auto_to_ABA_10um.min()
                        negative_mask = auto_to_ABA_10um < 0
                        if min_val < 0:
                            auto_to_ABA_10um[negative_mask] = auto_to_ABA_10um[negative_mask] * -1

                        mins = [np.percentile(auto_to_ABA_10um, 0.001), 0]
                        maxs = [np.percentile(auto_to_ABA_10um, 99.999), 3000]

                        for min, max, fmf in zip(mins, maxs, flatmap_figures):
                            auto_to_ABA_10um = skio.imread(auto_to_ABA_10um_path, plugin='simpleitk')
                            auto_to_ABA_10um[auto_to_ABA_10um < min] = min
                            auto_to_ABA_10um[auto_to_ABA_10um > max] = max

                            auto_to_ABA_10um = np.swapaxes(auto_to_ABA_10um, 0, 1)
                            auto_to_ABA_10um = np.swapaxes(auto_to_ABA_10um, 2, 1)
                            auto_to_ABA_10um = np.flip(auto_to_ABA_10um, 1)
                            # tifffile.imwrite(os.path.join(signal_to_reference_10um_directory, "result.tif"), auto_to_ABA_10um)
                            # auto_to_ABA_10um_n = tifffile.imread(os.path.join(signal_to_reference_10um_directory, "result.tif"))

                            # Normalize the array to the range 0-1
                            auto_to_ABA_10um_min = auto_to_ABA_10um.min()
                            auto_to_ABA_10um_max = auto_to_ABA_10um.max()
                            auto_to_ABA_10um_norm = (auto_to_ABA_10um - auto_to_ABA_10um_min) / \
                                                    (auto_to_ABA_10um_max - auto_to_ABA_10um_min)

                            normalized_layers = proj_butterfly_slab.project_volume(auto_to_ABA_10um_norm)

                            main_max = normalized_layers.max(axis=2).T
                            top_max = normalized_layers.max(axis=1).T
                            left_max = normalized_layers.max(axis=0)

                            main_shape = main_max.shape
                            top_shape = top_max.shape
                            left_shape = left_max.shape

                            # PLOT ALL LAYERS

                            # Set up a figure to plot them together
                            fig, axes = plt.subplots(2, 2,
                                                     gridspec_kw=dict(
                                                         width_ratios=(left_shape[1], main_shape[1]),
                                                         height_ratios=(top_shape[0], main_shape[0]),
                                                         hspace=0.01,
                                                         wspace=0.01),
                                                     figsize=(19.4, 12))

                            vmin = 0.0
                            vmax = 1.0
                            # Plot the surface view
                            axes[1, 1].imshow(main_max, vmin=vmin, vmax=vmax, cmap="magma", interpolation=None)

                            # plot our region boundaries
                            # for k, boundary_coords in bf_left_boundaries.items():
                            #     axes[1, 1].plot(*boundary_coords.T, c="white", lw=0.5)
                            # for k, boundary_coords in bf_right_boundaries.items():
                            #     axes[1, 1].plot(*boundary_coords.T, c="white", lw=0.5)

                            axes[1, 1].set(xticks=[], yticks=[], anchor="NW")

                            # Plot the top view
                            axes[0, 1].imshow(top_max, vmin=vmin, vmax=vmax, cmap="magma", interpolation=None)
                            axes[0, 1].set(xticks=[], yticks=[], anchor="SW")

                            # Plot the side view
                            axes[1, 0].imshow(left_max, vmin=vmin, vmax=vmax, cmap="magma", interpolation=None)
                            axes[1, 0].set(xticks=[], yticks=[], anchor="NE")

                            # Remove axes from unused plot area
                            axes[0, 0].set(xticks=[], yticks=[])
                            plt.savefig(fmf, dpi=300)

                        # PLOT LAYER BY LAYER

                        # # LAYER 2/3
                        # plt.figure()
                        # plt.imshow(
                        #     normalized_layers[:, :, top_l23:top_l4].max(axis=2).T,
                        #     vmin=0, vmax=1,
                        #     cmap="magma",
                        #     interpolation=None
                        # )
                        # # plot region boundaries
                        # for k, boundary_coords in bf_left_boundaries.items():
                        #     plt.plot(*boundary_coords.T, c="white", lw=0.5)
                        # for k, boundary_coords in bf_right_boundaries.items():
                        #     plt.plot(*boundary_coords.T, c="white", lw=0.5)
                        # plt.title("Layer 2/3")
                        # plt.savefig(os.path.join(signal_to_reference_10um_directory, "cortical_flatmap_layer_2-3.png"), dpi=300)
                        #
                        # # LAYER 4
                        # plt.figure()
                        # plt.imshow(
                        #     normalized_layers[:, :, top_l4:top_l5].max(axis=2).T,
                        #     vmin=0, vmax=1,
                        #     cmap="magma",
                        #     interpolation=None
                        # )
                        # # plot region boundaries
                        # for k, boundary_coords in bf_left_boundaries.items():
                        #     plt.plot(*boundary_coords.T, c="white", lw=0.5)
                        # for k, boundary_coords in bf_right_boundaries.items():
                        #     plt.plot(*boundary_coords.T, c="white", lw=0.5)
                        # plt.title("Layer 4")
                        # plt.savefig(os.path.join(signal_to_reference_10um_directory, "cortical_flatmap_layer_4.png"), dpi=300)

# def permute_data(img, ):
#     # permute
#     per = res.orientation_to_permuation(orientation)
#     img = img.transpose(per)
#
#     # reverse axes
#     re_slice = False
#     sl = [slice(None)] * img.ndim
#     for d, o in enumerate(orientation):
#         if o < 0:
#             sl[d] = slice(None, None, -1)
#             re_slice = True
#     if re_slice:
#         img = img[tuple(sl)]
#     return img

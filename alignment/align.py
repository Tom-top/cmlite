import os
import re
import sys
import tempfile
import shutil
import subprocess
import platform
import multiprocessing as mp

import numpy as np
from io import UnsupportedOperation

import utils.utils as ut
import utils.exceptions as excep

import IO.IO as io

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

        signal_to_reference_directory = os.path.join(sample_directory, f"signal_to_reference_{channel}")
        transform_atlas_parameter = dict(
            source=os.path.join(sample_directory, f"resampled_25um_{channel}.tif"),
            result_directory=signal_to_reference_directory,
            transform_parameter_file=os.path.join(auto_to_reference_directory, f"TransformParameters.1.txt"))
        if not os.path.exists(signal_to_reference_directory):
            ut.print_c(f"[INFO {sample_name}] Running signal to reference transform for channel {channel}!")
            transform_images(**transform_atlas_parameter)
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
            source=annotation_file,
            result_directory=atlas_to_auto_directory,
            transform_parameter_file=os.path.join(reference_to_auto_directory, f"TransformParameters_ni.1.txt"))
        if not os.path.exists(atlas_to_auto_directory):
            ut.print_c(f"[INFO {sample_name}] Running atlas to auto transform for channel {channel}!")
            transform_images(**transform_atlas_parameter)
        else:
            ut.print_c(f"[WARNING {sample_name}] Transforming: atlas to auto skipped for channel {channel}: "
                       f"atlas_to_auto_{channel} folder already exists!")

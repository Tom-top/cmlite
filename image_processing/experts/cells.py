import os
import shutil
import multiprocessing
import tempfile

import numpy as np
import numpy.lib.recfunctions as rfn
import gc
import cv2
import tifffile
import scipy.ndimage.filters as ndf

import utils.utils as ut

import IO.IO as io

import resampling.resampling as res

import alignment.align as elx
import alignment.annotation as ano

import parallel_processing.block_processing as bp
import parallel_processing.data_processing.array_processing as ap

import image_processing.illumination_correction as ic
import image_processing.filter.structure_element as se
import image_processing.filter.filter_kernel as fk
from image_processing.experts.utils import initialize_sinks, equalize, wrap_step, print_params

import analysis.measurements.shape_detection as sd
import analysis.measurements.maxima_detection as md
import analysis.measurements.measure_expression as me

import utils.timer as tmr

default_cell_detection_parameter = dict(
    # flatfield
    illumination_correction=None,

    # background removal
    background_correction=dict(shape=(10, 10),
                               form='Disk',
                               save=None),

    # equalization
    equalization=None,

    # difference of gaussians filter
    dog_filter=dict(shape=None,
                    sigma=None,
                    sigma2=None,
                    save=False),

    # extended maxima detection
    maxima_detection=dict(h_max=None,
                          shape=7,  # Neurons: 20; Microglia: 10
                          threshold=0,
                          valid=True,
                          save=None),

    # cell shape detection
    shape_detection=dict(threshold=700,
                         save=None),

    # cell intensity detection
    intensity_detection=dict(method='max',
                             shape=3,
                             measure=['source']),  # 'background_correction'
)

default_cell_detection_processing_parameter = dict(
    size_max=8,
    size_min=5,
    overlap=3,
    axes=[2],
    optimization=True,
    optimization_fix='all',
    verbose=True,
    processes=1,
)

# Fixme: Missing IDs in atlas -> probably background and ventricles but worth chercking
mouse_gubra_extra_labels = [(5576, 0, 'No label', 'NoL'),
                            (5577, 0, 'No label', 'NoL'),
                            (5767, 0, 'No label', 'NoL'),
                            (5768, 0, 'No label', 'NoL'),
                            (5769, 0, 'No label', 'NoL'),
                            (5770, 0, 'No label', 'NoL'),
                            (5771, 0, 'No label', 'NoL'),
                            (5822, 0, 'No label', 'NoL'),
                            (5826, 0, 'No label', 'NoL'),
                            (5844, 0, 'No label', 'NoL'),
                            (5860, 0, 'No label', 'NoL'),
                            (5873, 0, 'No label', 'NoL'),
                            (5874, 0, 'No label', 'NoL'),
                            (5875, 0, 'No label', 'NoL'),
                            (5876, 0, 'No label', 'NoL'),
                            (5877, 0, 'No label', 'NoL'),
                            (5878, 0, 'No label', 'NoL'),
                            (5879, 0, 'No label', 'NoL'),
                            (5880, 0, 'No label', 'NoL'),
                            (5912, 0, 'No label', 'NoL'),
                            ]


#
# def detect_cells(source, sink=None, cell_detection_parameter=default_cell_detection_parameter,
#                  processing_parameter=default_cell_detection_processing_parameter, workspace=None):
#     """Cell detection pipeline.
#
#     Arguments
#     ---------
#     source : source specification
#         The source of the stitched raw data.
#     sink : sink specification or None
#         The sink to write the result to. If None, an array is returned.
#     cell_detection_parameter : dict
#         Parameter for the binarization. See below for details.
#     processing_parameter : dict
#         Parameter for the parallel processing.
#         See :func:`ClearMap.ParallelProcessing.BlockProcessing.process` for
#         description of all the parameter.
#     workspace: Workspace
#         The optional workspace object to have a handle to cancel the multiprocess
#
#     Returns
#     -------
#     sink : Source
#         The result of the cell detection.
#
#     Notes
#     -----
#     Effectively this function performs the following steps:
#         * illumination correction via :func:`~ClearMap.ImageProcessing.IlluminationCorrection.correct_illumination`
#         * background removal
#         * difference of Gaussians (DoG) filter
#         * maxima detection via :func:`~ClearMap.Analysis.Measurements.MaximaDetection.find_extended_maxima`
#         * cell shape detection via :func:`~ClearMap.Analysis.Measurements.ShapeDetection.detect_shape`
#         * cell intensity and size measurements via: :func:`~ClearMap.ImageProcessing.Measurements.ShapeDetection.find_intensity`,
#           :func:`~ClearMap.ImageProcessing.Measurements.ShapeDetection.find_size`.
#
#
#     The parameters for each step are passed as sub-dictionaries to the
#     cell_detection_parameter dictionary.
#
#     * If None is passed for one of the steps this step is skipped.
#
#     * Each step also has an additional parameter 'save' that enables saving of
#     the result of that step to a file to inspect the pipeline.
#
#
#     Illumination correction
#     -----------------------
#     illumination_correction : dict or None
#         Illumination correction step parameter.
#
#         flatfield : array or str
#             The flat field estimate for the image planes.
#
#         background : array or None
#             A background level to assume for the flatfield correction.
#
#         scaling : float, 'max', 'mean' or None
#             Optional scaling after the flat field correction.
#
#         save : str or None
#             Save the result of this step to the specified file if not None.
#
#     See also :func:`ClearMap.ImageProcessing.IlluminationCorrection.correct_illumination`
#
#
#     Background removal
#     ------------------
#     background_correction : dict or None
#         Background removal step parameter.
#
#         shape : tuple
#             The shape of the structure element to estimate the background.
#             This should be larger than the typical cell size.
#
#         form : str
#             The form of the structure element (e.g. 'Disk')
#
#         save : str or None
#             Save the result of this step to the specified file if not None.
#
#     Equalization
#     ------------
#     equalization : dict or None
#         Equalization step parameter.
#         See also :func:`ClearMap.ImageProcessing.LocalStatistics.local_percentile`
#
#         precentile : tuple
#             The lower and upper percentiles used to estimate the equalization.
#             The lower percentile is used for normalization, the upper to limit the
#             maximal boost to a maximal intensity above this percentile.
#
#         max_value : float
#             The maximal intensity value in the equalized image.
#
#         selem : tuple
#             The structural element size to estimate the percentiles.
#             Should be larger than the larger vessels.
#
#         spacing : tuple
#             The spacing used to move the structural elements.
#             Larger spacings speed up processing but become locally less precise.
#
#         interpolate : int
#             The order of the interpolation used in constructing the full
#             background estimate in case a non-trivial spacing is used.
#
#         save : str or None
#           Save the result of this step to the specified file if not None.
#
#
#     DoG Filter
#     ----------
#     dog_filter : dict or None
#         Difference of Gaussian filter step parameter.
#
#         shape : tuple
#             The shape of the filter.
#             This should be near the typical cell size.
#
#         sigma : tuple or None
#              The std of the inner Gaussian.
#              If None, determined automatically from shape.
#
#         sigma2 : tuple or None
#              The std of the outer Gaussian.
#              If None, determined automatically from shape.
#
#         save : str or None
#             Save the result of this step to the specified file if not None.
#
#
#     Maxima detection
#     ----------------
#     maxima_detection : dict or None
#         Extended maxima detection step parameter.
#
#         h_max : float or None
#             The 'height' for the extended maxima.
#             If None, simple local maxima detection is used.
#
#         shape : tuple
#             The shape of the structural element for extended maxima detection.
#             This should be near the typical cell size.
#
#         threshold : float or None
#             Only maxima above this threshold are detected. If None, all maxima
#             are detected.
#
#         valid : bool
#             If True, only detect cell centers in the valid range of the blocks with
#             overlap.
#
#         save : str or None
#           Save the result of this step to the specified file if not None.
#
#
#     Shape detection
#     ---------------
#     shape_detection : dict or None
#         Shape detection step parameter.
#
#         threshold : float
#             Cell shape is expanded from maxima if pixels are above this threshold
#             and not closer to another maxima.
#
#         save : str or None
#           Save the result of this step to the specified file if not None.
#
#
#     Intensity detection
#     -------------------
#     intensity_detection : dict or None
#         Intensity detection step parameter.
#
#         method : {'max'|'min','mean'|'sum'}
#             The method to use to measure the intensity of a cell.
#
#         shape : tuple or None
#             If no cell shapes are detected a disk of this shape is used to measure
#             the cell intensity.
#
#         save : str or None
#             Save the result of this step to the specified file if not None.
#
#     References
#     ----------
#     [1] Renier, Adams, Kirst, Wu et al., "Mapping of Brain Activity by Automated Volume Analysis of Immediate Early Genes.", Cell 165, 1789 (2016)
#     [1] Kirst et al., "Mapping the Fine-Scale Organization and Plasticity of the Brain Vasculature", Cell 180, 780 (2020)
#     """
#
#     # initialize sink
#     shape = io.shape(source)
#     order = io.order(source)
#     print(cell_detection_parameter, shape, order)
#     initialize_sinks(cell_detection_parameter, shape, order)
#
#     cell_detection_parameter.update(verbose=processing_parameter.get('verbose', False))
#
#     n_processes = multiprocessing.cpu_count() if processing_parameter.get(
#         'processes') is None else processing_parameter.get('processes')
#     n_threads = int(multiprocessing.cpu_count() / n_processes)  # Number of threads so that * n_processes, fills CPUs
#
#     results, blocks = bp.process(detect_cells_block, source, sink=None, function_type='block', return_result=True,
#                                  return_blocks=True, parameter=cell_detection_parameter, workspace=workspace,
#                                  **{**processing_parameter, **{'n_threads': n_threads}})
#
#     # merge results
#     results = np.vstack([np.hstack(r) for r in results])
#
#     # create column headers  # FIXME: use pd.DataFrame instead
#     header = ['x', 'y', 'z']
#     dtypes = [int, int, int]
#     if cell_detection_parameter['shape_detection'] is not None:
#         header += ['size']
#         dtypes += [int]
#     measures = cell_detection_parameter['intensity_detection']['measure']
#     header += measures
#     dtypes += [float] * len(measures)
#
#     dt = {'names': header, 'formats': dtypes}
#     cells = np.zeros(len(results), dtype=dt)
#     for i, h in enumerate(header):
#         cells[h] = results[:, i]
#
#     # save results
#     return io.write(sink, cells)


def detect_cells(source, sink=None, cell_detection_parameter=default_cell_detection_parameter,
                 processing_parameter=default_cell_detection_processing_parameter, workspace=None):
    shape = io.shape(source)
    order = io.order(source)
    initialize_sinks(cell_detection_parameter, shape, order)

    cell_detection_parameter.update(verbose=processing_parameter.get('verbose', False))

    n_processes = multiprocessing.cpu_count() if processing_parameter.get(
        'processes') is None else processing_parameter.get('processes')
    n_threads = int(multiprocessing.cpu_count() / n_processes)  # Number of threads so that * n_processes, fills CPUs

    results, blocks = bp.process(detect_cells_block, source, sink=None, function_type='block', return_result=True,
                                 return_blocks=True, parameter=cell_detection_parameter, workspace=workspace,
                                 **{**processing_parameter, **{'n_threads': n_threads}})

    results = np.vstack([np.hstack(r) for r in results])

    header = ['x', 'y', 'z']
    dtypes = [int, int, int]
    if cell_detection_parameter['shape_detection'] is not None:
        header += ['size']
        dtypes += [int]
    measures = cell_detection_parameter['intensity_detection']['measure']
    header += measures
    dtypes += [float] * len(measures)

    dt = {'names': header, 'formats': dtypes}
    cells = np.zeros(len(results), dtype=dt)
    for i, h in enumerate(header):
        cells[h] = results[:, i]

    return io.write(sink, cells)


def detect_cells_block(source, parameter=default_cell_detection_parameter, n_threads=None):
    """Detect cells in a Block."""

    def initialize_worker():
        global md, sd, me, equalize, wrap_step, print_params, ic
        import analysis.measurements.maxima_detection as md
        import analysis.measurements.shape_detection as sd
        import analysis.measurements.measure_expression as me
        from image_processing.experts.utils import equalize, wrap_step, print_params
        import image_processing.illumination_correction as ic

    # Initialize worker-specific imports
    initialize_worker()

    # initialize parameter and slicing
    if parameter.get('verbose'):
        prefix = 'Block %s: ' % (source.info(),)
        total_time = tmr.Timer(prefix)
    else:
        prefix = ''

    base_slicing = source.valid.base_slicing
    valid_slicing = source.valid.slicing
    valid_lower = source.valid.lower
    valid_upper = source.valid.upper
    lower = source.lower

    steps_to_measure = {}  # FIXME: rename
    parameter_intensity = parameter.get('intensity_detection')
    if parameter_intensity:
        parameter_intensity = parameter_intensity.copy()
        measure = parameter_intensity.pop('measure', [])
        measure = measure if measure else []
        valid_measurement_keys = list(default_cell_detection_parameter.keys()) + ['source']
        for m in measure:
            if m not in valid_measurement_keys:
                raise KeyError(f'Unknown measurement: {m}')
            steps_to_measure[m] = None

    if 'source' in steps_to_measure:
        steps_to_measure['source'] = source

    default_step_params = {'parameter': parameter, 'steps_to_measure': steps_to_measure, 'prefix': prefix,
                           'base_slicing': base_slicing, 'valid_slicing': valid_slicing}

    # WARNING: if param_illumination: previous_step = source, not np.array(source.array)
    corrected = wrap_step('illumination_correction', np.array(source.array),
                          ic.correct_illumination, **default_step_params)

    background = wrap_step('background_correction', corrected, remove_background,
                           remove_previous_result=True, **default_step_params)

    equalized = wrap_step('equalization', background, equalize, remove_previous_result=True,
                          extra_kwargs={'mask': None}, **default_step_params)

    dog = wrap_step('dog_filter', equalized, dog_filter,  # TODO: DoG filter != .title()
                    remove_previous_result=True, **default_step_params)

    # Maxima detection
    parameter_maxima = parameter.get('maxima_detection')
    parameter_shape = parameter.get('shape_detection')

    if parameter_shape or parameter_intensity:
        if not parameter_maxima:
            print(f'{prefix}Warning: maxima detection needed for shape and intensity detection!')
            parameter_maxima = {}

    if parameter_maxima:
        valid = parameter_maxima.pop('valid', None)
        maxima = wrap_step('maxima_detection', dog, md.find_maxima, extra_kwargs={'verbose': parameter.get('verbose')},
                           remove_previous_result=False, **default_step_params)
        # center of maxima
        if parameter_maxima['h_max']:  # FIXME: check if source or dog
            centers = md.find_center_of_maxima(source, maxima=maxima, verbose=parameter.get('verbose'))
        else:
            centers = ap.where(maxima, processes=n_threads,
                               cutoff=np.inf).array  # Fixme: parallel processing is broken: np.inf to prevent error
        del maxima

        # correct for valid region
        if valid:
            ids = np.ones(len(centers), dtype=bool)
            for c, l, u in zip(centers.T, valid_lower, valid_upper):
                ids = np.logical_and(ids, np.logical_and(l <= c, c < u))
            centers = centers[ids]
            del ids
        results = (centers,)
    else:
        results = ()
    # print(f"Centers filt: {centers}")

    # WARNING: sd.detect_shape uses prange
    # cell shape detection  # FIXME: may use centers without assignment
    shape = wrap_step('shape_detection', dog, sd.detect_shape, remove_previous_result=True, **default_step_params,
                      args=[centers], extra_kwargs={'verbose': parameter.get('verbose'), 'processes': n_threads})
    if parameter_shape:
        # size detection
        max_label = centers.shape[0]
        sizes = sd.find_size(shape, max_label=max_label)
        valid = sizes > 0

        results += (sizes,)
    else:
        valid = None
        shape = None

    # cell intensity detection
    if parameter_intensity:
        parameter_intensity, timer = print_params(parameter_intensity, 'intensity_detection', prefix,
                                                  parameter.get('verbose'))

        if shape is not None:
            r = parameter_intensity.pop('shape', 3)
            if isinstance(r, tuple):
                r = r[0]

        for m in measure:
            if shape is not None:
                intensity = sd.find_intensity(steps_to_measure[m], label=shape,
                                              max_label=max_label, **parameter_intensity)
            else:  # WARNING: prange but me.measure_expression not parallel since processes=1
                intensity = me.measure_expression(steps_to_measure[m], centers, search_radius=r,
                                                  **parameter_intensity, processes=1, verbose=False)
            results += (intensity,)

        if parameter.get('verbose'):
            timer.print_elapsed_time('Shape detection')

    if valid is not None:
        results = tuple(r[valid] for r in results)
    # correct coordinate offsets of blocks
    results = (results[0] + lower,) + results[1:]
    # correct shapes for merging
    results = tuple(r[:, None] if r.ndim == 1 else r for r in results)

    if parameter.get('verbose'):
        total_time.print_elapsed_time('Cell detection')

    gc.collect()

    return results


def remove_background(source, shape, form='Disk'):
    selem = se.structure_element(shape, form=form, ndim=2)  # FIXME: use skimage kernel
    selem = np.array(selem).astype('uint8')
    removed = np.empty(source.shape, dtype=source.dtype)
    for z in range(source.shape[2]):
        removed[:, :, z] = source[:, :, z] - np.minimum(source[:, :, z],
                                                        cv2.morphologyEx(source[:, :, z], cv2.MORPH_OPEN, selem))
    return removed


def dog_filter(source, shape, sigma=None, sigma2=None):
    if shape is not None:
        fdog = fk.filter_kernel(ftype='dog', shape=shape, sigma=sigma, sigma2=sigma2)
        fdog = fdog.astype('float32')
        filtered = ndf.correlate(source, fdog)
        filtered[filtered < 0] = 0
        return filtered
    else:
        return source


def detect_maxima(source, h_max=None, shape=5, threshold=None, verbose=False):  # FIXME: use to refactor
    # extended maxima
    maxima = md.find_maxima(source, h_max=h_max, shape=shape, threshold=threshold, verbose=verbose)

    # center of maxima
    if h_max:
        centers = md.find_center_of_maxima(source, maxima=maxima, verbose=verbose)
    else:
        centers = ap.where(maxima).array  # FIXME: prange

    return centers


def transformation(sample_directory, channel, coordinates, atlas):
    coordinates = res.resample_points(coordinates, sink=None, orientation=None,
                                      source_shape=io.shape(os.path.join(sample_directory, f"stitched_{channel}.npy")),
                                      sink_shape=io.shape(
                                          os.path.join(sample_directory, f"resampled_25um_{channel}.tif")),
                                      )
    elx.write_points(os.path.join(elx.elastix_output_folder, "outputpoints.txt"), coordinates, indices=False,
                     binary=False)

    auto_to_signal_directory = os.path.join(sample_directory, f'auto_to_signal_{channel}')
    coordinates, _ = elx.transform_points_with_transformix(
        os.path.join(elx.elastix_output_folder, "outputpoints.txt"),
        elx.elastix_output_folder,
        os.path.join(auto_to_signal_directory, "TransformParameters.0.txt"),
        transformix_input=False,
    )

    reference_to_auto_directory = os.path.join(sample_directory, f'{atlas}_reference_to_auto_{channel}')
    coordinates, _ = elx.transform_points_with_transformix(
        os.path.join(elx.elastix_output_folder, "outputpoints.txt"),
        elx.elastix_output_folder,
        os.path.join(reference_to_auto_directory, "TransformParameters.0.txt"),
        transformix_input=True,
    )
    coordinates, _ = elx.transform_points_with_transformix(
        os.path.join(elx.elastix_output_folder, "outputpoints.txt"),
        elx.elastix_output_folder,
        os.path.join(reference_to_auto_directory, "TransformParameters.1.txt"),
        transformix_input=True,
    )

    return coordinates


def filter_cells(source, sink, thresholds):
    """
    Filter an array of detected cells according to the thresholds.

    Arguments
    ---------
    source : str, array or Source
        The source for the cell data.
    sink : str, array or Source
        The sink for the results.
    thresholds : dict
        Dictionary of the form {name : threshold} where name refers to the
        column in the cell data and threshold can be None, a float
        indicating a minimal threshold or a tuple (min,max) where min,max can be
        None or a minimal and maximal threshold value.

    Returns
    -------
    sink : str, array or Source
        The thresholded cell data.
    """
    source = io.as_source(source)

    ids = np.ones(source.shape[0], dtype=bool)
    for filter_name, thrsh in thresholds.items():
        if thrsh:
            if not isinstance(thrsh, (tuple, list)):
                thrsh = (thrsh, None)
            if thrsh[0] is not None:
                ids = np.logical_and(ids, thrsh[0] <= source[filter_name])
            if thrsh[1] is not None:
                ids = np.logical_and(ids, thrsh[1] > source[filter_name])
    cells_filtered = source[ids]

    return io.write(sink, cells_filtered)


def segment_cells(sample_name, sample_directory, annotation_files, analysis_data_size_directory,
                  data_to_segment=None, save_segmented_cells=True,
                  **kwargs):
    print("")
    for channel in kwargs["study_params"]["channels_to_segment"]:
        p = kwargs["cell_detection"]
        shape_detection_directory = os.path.join(sample_directory, f"shape_detection_{p['shape_detection']}")
        if not os.path.exists(shape_detection_directory):
            os.mkdir(shape_detection_directory)

        cell_detection_parameter = default_cell_detection_parameter.copy()
        cell_detection_parameter['shape_detection']['threshold'] = p['shape_detection']

        ################################################################################################################
        # 3.1 DETECT CELLS
        ################################################################################################################

        cells_raw_path = os.path.join(shape_detection_directory, f"cells_raw_{channel}.npy")
        if not os.path.exists(cells_raw_path):
            if p['save_int_results']:
                cell_detection_parameter['background_correction']['save'] = os.path.join(shape_detection_directory,
                                                                                         "background_removal.tif")
                if os.path.exists(cell_detection_parameter['background_correction']['save']):
                    os.remove(cell_detection_parameter['background_correction']['save'])
                cell_detection_parameter['maxima_detection']['save'] = os.path.join(shape_detection_directory,
                                                                                    "maxima_detection.tif")
                if os.path.exists(cell_detection_parameter['maxima_detection']['save']):
                    os.remove(cell_detection_parameter['maxima_detection']['save'])
                cell_detection_parameter['shape_detection']['save'] = os.path.join(shape_detection_directory,
                                                                                   "shape_detection.tif")
                if os.path.exists(cell_detection_parameter['shape_detection']['save']):
                    os.remove(cell_detection_parameter['shape_detection']['save'])
            ut.print_c(f"[INFO {sample_name}] Running cell detection for channel {channel}!")
            if data_to_segment is None:
                detect_cells(os.path.join(sample_directory, f"stitched_{channel}.npy"),
                             cells_raw_path,
                             cell_detection_parameter=cell_detection_parameter,
                             processing_parameter=default_cell_detection_processing_parameter)
            else:
                detect_cells(data_to_segment,
                             cells_raw_path,
                             cell_detection_parameter=cell_detection_parameter,
                             processing_parameter=default_cell_detection_processing_parameter)
        else:
            ut.print_c(f"[WARNING {sample_name}] Skipping cell detection for channel {channel}: "
                       f"cells_raw_{channel}.npy file already exists!")

        ################################################################################################################
        # 3.2 FILTER CELLS
        ################################################################################################################

        cells_filtered_path = os.path.join(shape_detection_directory, f"cells_filtered_{channel}.npy")
        if not os.path.exists(cells_filtered_path):
            ut.print_c(f"[INFO {sample_name}] Filtering cells for channel {channel}!")
            filter_cells(source=cells_raw_path,
                         sink=cells_filtered_path,
                         thresholds=p["thresholds"])
        else:
            ut.print_c(f"[WARNING {sample_name}] Skipping cell filtering for channel {channel}: "
                       f"cells_filtered_{channel}.npy file already exists!")

        analysis_sample_directory = os.path.join(analysis_data_size_directory, sample_name)
        if not os.path.exists(analysis_sample_directory):
            os.mkdir(analysis_sample_directory)
        shutil.copyfile(cells_filtered_path,
                        os.path.join(analysis_sample_directory, f"cells_filtered_{channel}.npy"))

        cells_filtered = io.as_source(cells_filtered_path)
        filetered_cell_coordinates = np.array([cells_filtered[c] for c in 'xyz']).T

        ################################################################################################################
        # [OPTIONAL] DISPLAY SEGMENTED CELLS IN 10um ISOTROPIC
        ################################################################################################################

        if save_segmented_cells:
            labeled_cells_10um_path = os.path.join(sample_directory, f"labeled_cells_10um_{channel}.tif")
            if not os.path.exists(labeled_cells_10um_path):
                ut.print_c(f"[INFO {sample_name}] Drawing segmented cells for channel {channel}!")
                resampled_signal_10um_path = os.path.join(sample_directory, f"resampled_10um_{channel}.tif")
                filetered_cell_coordinates_10um = res.resample_points(
                    filetered_cell_coordinates, sink=None, orientation=None,
                    source_shape=io.shape(os.path.join(sample_directory, f"stitched_{channel}.npy")),
                    sink_shape=io.shape(resampled_signal_10um_path)
                )
                resampled_signal_10um = tifffile.imread(resampled_signal_10um_path)
                resampled_signal_10um = np.repeat(resampled_signal_10um[..., np.newaxis], 3, axis=-1)
                filetered_cell_coordinates_10um[:, [0, 2]] = filetered_cell_coordinates_10um[:, [2, 0]]
                for x, y, z in filetered_cell_coordinates_10um.astype(int):
                    resampled_signal_10um[x, y, z] = [2 ** 16 - 1, 0, 0]
                tifffile.imwrite(labeled_cells_10um_path, resampled_signal_10um)
            else:
                ut.print_c(f"[WARNING {sample_name}] Skipping drawing of segmented cells for channel {channel}: "
                           f"labeled_cells_10um_{channel}.tif file already exists!")

        for atlas, annotation in zip(kwargs['study_params']['atlas_to_use'], annotation_files):

            ############################################################################################################
            # 3.3 TRANSFORM CELLS
            ############################################################################################################

            atlas_name = atlas.split('_')[-1]
            animal_species = atlas.split('_')[0]

            cells_transformed_path = os.path.join(shape_detection_directory,
                                                  f"{atlas}_cells_transformed_{channel}.npy")
            if not os.path.exists(cells_transformed_path):
                ut.print_c(f"[INFO {sample_name}] Transforming cells for channel {channel}!")
                transformed_cell_coordinates = transformation(sample_directory, channel, filetered_cell_coordinates,
                                                              atlas)
                if atlas_name == "gubra":
                    ano.set_annotation_file(annotation,
                                            label_file=f"resources/atlas/{atlas_name}_annotation_"
                                                       f"{animal_species}.json",
                                            extra_label=mouse_gubra_extra_labels)
                else:
                    ano.set_annotation_file(annotation,
                                            f"resources/atlas/{atlas_name}_annotation_{animal_species}.json")
                label = ano.label_points(transformed_cell_coordinates, key='order')
                names = ano.convert_label(label, key='order', value='name')
                transformed_cell_coordinates.dtype = [(t, float) for t in ('xt', 'yt', 'zt')]
                label = np.array(label, dtype=[('order', int)])
                names = np.array(names, dtype=[('name', 'U256')])

                cells_data = rfn.merge_arrays([cells_filtered[:], transformed_cell_coordinates, label, names],
                                              flatten=True, usemask=False)
                io.write(cells_transformed_path, cells_data)
                shutil.copyfile(cells_transformed_path,
                                os.path.join(analysis_sample_directory, f"{atlas}_cells_transformed_{channel}.npy"))
            else:
                ut.print_c(f"[WARNING {sample_name}] Skipping transform cells for channel {channel}: "
                           f"cells_transformed_{atlas_name}_{channel}.npy file already exists!")

            ############################################################################################################
            # 3.4 WRITE RESULT
            ############################################################################################################

            cells_transformed_csv_path = os.path.join(shape_detection_directory,
                                                      f"{atlas}_cells_transformed_{channel}.csv")
            cells_transformed = io.as_source(cells_transformed_path)
            if not os.path.exists(cells_transformed_csv_path):
                ut.print_c(f"[INFO {sample_name}] Saving cell counts as csv for channel {channel}!")
                delimiter = ";"
                header = f'{delimiter}'.join([h for h in cells_transformed.dtype.names])
                np.savetxt(cells_transformed_csv_path, cells_transformed[:], header=header,
                           delimiter=delimiter,
                           fmt='%d;%d;%d;%d;%.6f;%.6f;%.6f;%.6f;%d;"%s"',
                           comments="")
                shutil.copyfile(cells_transformed_csv_path,
                                os.path.join(analysis_sample_directory, f"{atlas}_cells_transformed_{channel}.csv"))
            else:
                ut.print_c(f"[WARNING {sample_name}] Skipping saving cell counts as csv for channel {channel}: "
                           f"{atlas}_cells_transformed_{channel}.csv file already exists!")

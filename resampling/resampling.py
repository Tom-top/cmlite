import os
import tempfile
import itertools

import json
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
from natsort import natsorted
import functools as ft
import numpy as np
import cv2

import IO.IO as io
import IO.file_list as fl
import IO.file_utils as fu

import utils.utils as ut

import parallel_processing.process_writer as pw
import parallel_processing.parallel_traceback as ptb


def format_orientation(orientation, inverse=False, default=None):
    """Convert orientation to standard format.

    Arguments
    ---------
    orientation : tuple or str
      The orientation specification.
    inverse : bool
       If True, invert orientation.
    default : object
       The default value if orientation is None

    Returns
    -------
    orientation : tuple of ints
      The orientation sequence.

    See Also
    --------
    `Orientation`_
    """
    if orientation is None:
        return default

    # fix named representations
    if orientation == 'left':
        # orientation = (1,2,3)
        orientation = None
    elif orientation == 'right':
        orientation = (-1, 2, 3)

    if orientation is not None and len(orientation) != 3:
        raise ValueError("orientation should be 'left', 'right' or a tuple of 3 (signed) "
                         "integers from 1 to 3, found %r" % (orientation,))

    if inverse:
        orientation = inverse_orientation(orientation)

    return orientation


def inverse_orientation(orientation):
    """Returns the inverse orientation taking axis inversions into account.

    Arguments
    ---------
    orientation : tuple or str
      The orientation specification.

    Returns
    -------
    orientation : tuple
      The inverse orientation sequence.

    See Also
    --------
    `Orientation`_
    """
    orientation = format_orientation(orientation)

    if orientation is None:
        return None

    # orientation is defined as permuting the axes and then inverrting the axis
    inv = list(orientation)
    for i, o in enumerate(orientation):
        if o < 0:
            inv[int(abs(o) - 1)] = -(i + 1)
        else:
            inv[int(abs(o) - 1)] = (i + 1)

    return tuple(inv)


def resample_shape(source_shape, sink_shape=None, source_resolution=None, sink_resolution=None, orientation=None):
    """Calculate scaling factors and data shapes for resampling.

    Arguments
    ---------
    source_shape : tuple
      The shape the source.
    sink_shape : tuple or None
      The shape of the resmapled sink.
    source_resolution : tuple or None
      The resolution of the source.
    sink_resolution : tuple or None
      The resolution of the sink.
    orientation : tuple or str
      The re-orientation specification.

    Returns
    -------
    source_shape : tuple
      The shape of the source.
    sink_shape : tuple
      The shape of the sink.
    source_resolution : tuple or None
      The resolution of the source.
    sink_resolution : tuple or None
      The resolution of the sink.

    See Also
    --------
    `Orientation`_
    """
    orientation = format_orientation(orientation)

    # determine shapes if not specified
    if sink_shape is None and source_shape is None:
        raise RuntimeError('Source or sink shape must be defined!')

    if sink_shape is None and sink_resolution is None:
        raise RuntimeError('Sink shape or resolution must be defined!')

    if sink_resolution is None and not isinstance(sink_shape, tuple):
        sink_shape = io.shape(sink_shape)

    if source_shape is not None:
        ndim = len(source_shape)
    else:
        ndim = len(sink_shape)

    if sink_shape is None:
        if source_resolution is None:
            source_resolution = (1.0,) * ndim
        if sink_resolution is None:
            sink_resolution = (1.0,) * ndim

        # orient resolution of source to resolution of sink to get sink shape
        source_resolutionO = orient_resolution(source_resolution, orientation)
        source_shapeO = orient_shape(source_shape, orientation)
        sink_shape = tuple(
            [int(np.ceil(source_shapeO[i] * float(source_resolutionO[i]) / float(sink_resolution[i]))) for i in
             range(ndim)])

    if source_shape is None:
        if source_resolution is None:
            source_resolution = (1,) * ndim
        if sink_resolution is None:
            sink_resolution = (1,) * ndim

        # orient resolution of source to resolution of sink to get sink shape
        source_resolutionO = orient_resolution(source_resolution, orientation)
        source_shape = tuple(
            [int(np.ceil(sink_shape[i] * float(sink_resolution[i]) / float(source_resolutionO[i]))) for i in
             range(ndim)])
        source_shape = orient_shape(source_shape, orientation, inverse=True)

    # calculate effecive resolutions
    if source_resolution is None:
        if sink_resolution is None:
            source_resolution = (1, 1, 1)
        else:
            source_shapeO = orient_shape(source_shape, orientation)
            source_resolution = tuple(
                float(sink_shape[i]) / float(source_shapeO[i]) * sink_resolution[i] for i in range(ndim))
            source_resolution = orient_resolution(source_resolution, orientation, inverse=True)

    source_shapeO = orient_shape(source_shape, orientation)
    source_resolutionO = orient_resolution(source_resolution, orientation)
    sink_resolution = tuple(
        float(source_shapeO[i]) / float(sink_shape[i]) * source_resolutionO[i] for i in range(ndim))

    return source_shape, sink_shape, source_resolution, sink_resolution


def orient_shape(shape, orientation, inverse=False):
    """Permutes a shape according to the given orientation.

    Arguments
    ---------
    shape : tuple
      The shape specification.
    orientation : tuple or str
      The orientation specification.
    inverse : bool
      If True, invert the orientation.

    Returns
    -------
    shape : tuple
      The oriented shape tuple.

    See Also
    --------
    `Orientation`_
    """
    return orient_resolution(shape, orientation, inverse=inverse)


def orient_resolution(resolution, orientation, inverse=False):
    """Permutes a resolution tuple according to the given orientation.

    Arguments
    ---------
      resolution : tuple
        The resolution specification.
      orientation : tuple or str
        The orientation specification.
      inverse : bool
        If True, invert the orientation.

    Returns
    -------
      resolution : tuple
        The re-oriented resolution sequence.

    See Also
    --------
    `Orientation`_
    """
    if resolution is None:
        return None

    per = orientation_to_permuation(orientation, inverse=inverse)

    return tuple(resolution[i] for i in per)


def orientation_to_permuation(orientation, inverse=False):
    """Extracts the permuation from an orientation.

    Arguments
    ---------
    orientation : tuple or str
      The orientation specification.
    inverse : bool
      If True, return inverse permutation.

    Returns
    -------
    permuation : tuple of ints
      The premutation sequence.

    See Also
    --------
    `Orientation`_
    """
    orientation = format_orientation(orientation, inverse=inverse)
    if orientation is None:
        return (0, 1, 2)
    else:
        return tuple(int(abs(i)) - 1 for i in orientation)


@ptb.parallel_traceback
def _resample_2d(index, source, sink, axes, shape, interpolation, n_indices, verbose):
    """Resampling helper function to use for parallel resampling of image slices"""
    if verbose:
        pw.ProcessWriter(index).write(f"Resampling: Axes {axes}, slice {index[0]}/{n_indices}")

    # slicing
    ndim = len(shape)
    slicing_ = ()
    i = 0
    for d in range(ndim):
        if d in axes:
            slicing_ += (slice(None),)
        else:
            slicing_ += (index[i],)
            i += 1

    # resample planeresizeresizeresize
    sink = sink.as_real()
    source = source.as_real()
    new_shape = (shape[axes[1]], shape[axes[0]])
    sink[slicing_] = cv2.resize(source[slicing_], new_shape, interpolation=interpolation)
    # WARNING: cv2 takes reverse shape order !


def _axes_order(axes_order, source, sink_shape_in_source_orientation, order=None):
    """Helper to find axes order for subsequent 2d resampling steps."""

    source_shape = source.shape
    ndim = source.ndim

    if axes_order is not None and isinstance(axes_order, list):
        axes_order = [(a[0], a[1]) if a[0] < a[1] else (a[1], a[0]) for a in axes_order]
        shape_order = []
        last_shape = source_shape
        for axes in axes_order:
            if not isinstance(axes, tuple) and len(axes) != 2:
                raise ValueError('resampling expected a tuple of len 2 for axes_order entry, got %r!' % axes)
            last_shape = tuple([s if d not in axes else t for d, s, t in
                                zip(range(ndim), last_shape, sink_shape_in_source_orientation)])
            shape_order.append(last_shape)
        return axes_order, shape_order

    else:  # determine automatically
        if axes_order is None:
            axes_order = 'order'
        if axes_order == 'order' and order is None and not isinstance(source, fl.Source):
            axes_order = 'size'

        if axes_order == 'size':  # order to reduce size as much as possible in each sub-resampling step
            resample_axes = np.array(
                [d for d, s, t in zip(range(ndim), sink_shape_in_source_orientation, source_shape) if s != t])
            resample_factors = np.array(
                [float(t) / float(s) for s, t in zip(sink_shape_in_source_orientation, source_shape) if s != t])

            axes_order = []
            shape_order = []
            last_shape = source_shape

            while len(resample_axes) > 0:
                if len(resample_axes) >= 2:
                    # take largest two resampling factors
                    ids = np.argsort(resample_factors)[-2:]
                    axes = tuple(np.sort(resample_axes[ids]))
                    last_shape = tuple([s if d not in axes else t for d, s, t in
                                        zip(range(ndim), last_shape, sink_shape_in_source_orientation)])

                    axes_order.append(axes)
                    shape_order.append(last_shape)

                    resample_axes = np.array([s for a, s in enumerate(resample_axes) if a not in ids])
                    resample_factors = np.array([s for a, s in enumerate(resample_factors) if a not in ids])

                else:
                    axis = resample_axes[0]
                    small_axis = np.argsort(last_shape)
                    small_axis = [a for a in small_axis if a != axis][0]
                    if axis < small_axis:
                        axes = (axis, small_axis)
                    else:
                        axes = (small_axis, axis)
                    last_shape = tuple([s if d not in axes else t for d, s, t in
                                        zip(range(ndim), last_shape, sink_shape_in_source_orientation)])

                    axes_order.append(axes)
                    shape_order.append(last_shape)

                    resample_axes = []

            return axes_order, shape_order

        elif axes_order == 'order':  # order axes according to array order for faster io

            if isinstance(source, fl.Source):
                # FileList determine order according to file structure
                axes_list = source.axes_list
                # axes_file = source.axes_file

                resample_axes = np.array(
                    [d for d, s, t in zip(range(ndim), sink_shape_in_source_orientation, source_shape) if s != t])
                resample_factors = np.array(
                    [float(t) / float(s) for s, t in zip(sink_shape_in_source_orientation, source_shape) if s != t])

                axes_order = []
                shape_order = []
                last_shape = source_shape

                # modify factors to account for file structure
                resample_factors_list = [f for a, f in zip(resample_axes, resample_factors) if a in axes_list]
                if len(resample_factors_list) > 0:
                    max_resample_factor_list = np.max(resample_factors_list)
                else:
                    max_resample_factor_list = 0
                resample_factors_sort = np.array([f if a in axes_list else f + max_resample_factor_list for a, f in
                                                  zip(resample_axes, resample_factors)])
                # print(resample_factors_sort, resample_factors)

                while len(resample_axes) > 0:
                    if len(resample_axes) >= 2:
                        ids = np.argsort(resample_factors_sort)[-2:]

                        axes = tuple(np.sort(resample_axes[ids]))
                        last_shape = tuple([s if d not in axes else t for d, s, t in
                                            zip(range(ndim), last_shape, sink_shape_in_source_orientation)])

                        axes_order.append(axes)
                        shape_order.append(last_shape)

                        resample_axes = np.array([s for a, s in enumerate(resample_axes) if a not in ids])
                        resample_factors = np.array([s for a, s in enumerate(resample_factors) if a not in ids])
                        resample_factors_sort = np.array(
                            [s for a, s in enumerate(resample_factors_sort) if a not in ids])
                    else:
                        axis = resample_axes[0]
                        small_axis = np.argsort(last_shape)
                        small_axis = [a for a in small_axis if a != axis][0]
                        if axis < small_axis:
                            axes = (axis, small_axis)
                        else:
                            axes = (small_axis, axis)
                        last_shape = tuple([s if d not in axes else t for d, s, t in
                                            zip(range(ndim), last_shape, sink_shape_in_source_orientation)])

                        axes_order.append(axes)
                        shape_order.append(last_shape)

                        resample_axes = []

                return axes_order, shape_order
            else:
                # not a FileList
                resample_axes = np.array(
                    [d for d, s, t in zip(range(ndim), sink_shape_in_source_orientation, source_shape) if s != t])

                axes_order = []
                shape_order = []
                last_shape = source_shape
                while len(resample_axes) > 0:
                    if len(resample_axes) >= 2:
                        if order == 'C':
                            slicing = slice(-2, None)
                        else:
                            slicing = slice(None, 2)
                        axes = tuple(resample_axes[slicing])
                    else:
                        if order == 'C':
                            axes = (resample_axes[0], axes_order[-1][0])
                        else:
                            axes = (axes_order[-1][1], resample_axes[0])

                    axes_order.append(axes)
                    last_shape = tuple([s if d not in axes else t for d, s, t in
                                        zip(range(ndim), last_shape, sink_shape_in_source_orientation)])
                    shape_order.append(last_shape)
                    resample_axes = np.array([a for a in resample_axes if a not in axes])

                return axes_order, shape_order

        else:
            raise ValueError("axes_order %r not 'size','order' or list but %r!" % axes_order)


def _interpolation_to_cv2(interpolation):
    """Helper to convert interpolation specification to CV2 format."""

    if interpolation in ['nearest', 'nn', None, cv2.INTER_NEAREST]:
        interpolation = cv2.INTER_NEAREST
    elif interpolation in ['area', 'a', cv2.INTER_AREA]:
        interpolation = cv2.INTER_AREA
    else:
        interpolation = cv2.INTER_LINEAR

    return interpolation


def _axes_order(axes_order, source, sink_shape_in_source_orientation, order=None):
    """Helper to find axes order for subsequent 2d resampling steps."""

    source_shape = source.shape
    ndim = source.ndim

    if axes_order is not None and isinstance(axes_order, list):
        axes_order = [(a[0], a[1]) if a[0] < a[1] else (a[1], a[0]) for a in axes_order]
        shape_order = []
        last_shape = source_shape
        for axes in axes_order:
            if not isinstance(axes, tuple) and len(axes) != 2:
                raise ValueError('resampling expected a tuple of len 2 for axes_order entry, got %r!' % axes)
            last_shape = tuple([s if d not in axes else t for d, s, t in
                                zip(range(ndim), last_shape, sink_shape_in_source_orientation)])
            shape_order.append(last_shape)
        return axes_order, shape_order

    else:  # determine automatically
        if axes_order is None:
            axes_order = 'order'
        if axes_order == 'order' and order is None and not isinstance(source, fl.Source):
            axes_order = 'size'

        if axes_order == 'size':  # order to reduce size as much as possible in each sub-resampling step
            resample_axes = np.array(
                [d for d, s, t in zip(range(ndim), sink_shape_in_source_orientation, source_shape) if s != t])
            resample_factors = np.array(
                [float(t) / float(s) for s, t in zip(sink_shape_in_source_orientation, source_shape) if s != t])

            axes_order = []
            shape_order = []
            last_shape = source_shape

            while len(resample_axes) > 0:
                if len(resample_axes) >= 2:
                    # take largest two resampling factors
                    ids = np.argsort(resample_factors)[-2:]
                    axes = tuple(np.sort(resample_axes[ids]))
                    last_shape = tuple([s if d not in axes else t for d, s, t in
                                        zip(range(ndim), last_shape, sink_shape_in_source_orientation)])

                    axes_order.append(axes)
                    shape_order.append(last_shape)

                    resample_axes = np.array([s for a, s in enumerate(resample_axes) if a not in ids])
                    resample_factors = np.array([s for a, s in enumerate(resample_factors) if a not in ids])

                else:
                    axis = resample_axes[0]
                    small_axis = np.argsort(last_shape)
                    small_axis = [a for a in small_axis if a != axis][0]
                    if axis < small_axis:
                        axes = (axis, small_axis)
                    else:
                        axes = (small_axis, axis)
                    last_shape = tuple([s if d not in axes else t for d, s, t in
                                        zip(range(ndim), last_shape, sink_shape_in_source_orientation)])

                    axes_order.append(axes)
                    shape_order.append(last_shape)

                    resample_axes = []

            return axes_order, shape_order

        elif axes_order == 'order':  # order axes according to array order for faster io

            if isinstance(source, fl.Source):
                # FileList determine order according to file structure
                axes_list = source.axes_list
                # axes_file = source.axes_file

                resample_axes = np.array(
                    [d for d, s, t in zip(range(ndim), sink_shape_in_source_orientation, source_shape) if s != t])
                resample_factors = np.array(
                    [float(t) / float(s) for s, t in zip(sink_shape_in_source_orientation, source_shape) if s != t])

                axes_order = []
                shape_order = []
                last_shape = source_shape

                # modify factors to account for file structure
                resample_factors_list = [f for a, f in zip(resample_axes, resample_factors) if a in axes_list]
                if len(resample_factors_list) > 0:
                    max_resample_factor_list = np.max(resample_factors_list)
                else:
                    max_resample_factor_list = 0
                resample_factors_sort = np.array([f if a in axes_list else f + max_resample_factor_list for a, f in
                                                  zip(resample_axes, resample_factors)])
                # print(resample_factors_sort, resample_factors)

                while len(resample_axes) > 0:
                    if len(resample_axes) >= 2:
                        ids = np.argsort(resample_factors_sort)[-2:]

                        axes = tuple(np.sort(resample_axes[ids]))
                        last_shape = tuple([s if d not in axes else t for d, s, t in
                                            zip(range(ndim), last_shape, sink_shape_in_source_orientation)])

                        axes_order.append(axes)
                        shape_order.append(last_shape)

                        resample_axes = np.array([s for a, s in enumerate(resample_axes) if a not in ids])
                        resample_factors = np.array([s for a, s in enumerate(resample_factors) if a not in ids])
                        resample_factors_sort = np.array(
                            [s for a, s in enumerate(resample_factors_sort) if a not in ids])
                    else:
                        axis = resample_axes[0]
                        small_axis = np.argsort(last_shape)
                        small_axis = [a for a in small_axis if a != axis][0]
                        if axis < small_axis:
                            axes = (axis, small_axis)
                        else:
                            axes = (small_axis, axis)
                        last_shape = tuple([s if d not in axes else t for d, s, t in
                                            zip(range(ndim), last_shape, sink_shape_in_source_orientation)])

                        axes_order.append(axes)
                        shape_order.append(last_shape)

                        resample_axes = []

                return axes_order, shape_order
            else:
                # not a FileList
                resample_axes = np.array(
                    [d for d, s, t in zip(range(ndim), sink_shape_in_source_orientation, source_shape) if s != t])

                axes_order = []
                shape_order = []
                last_shape = source_shape
                while len(resample_axes) > 0:
                    if len(resample_axes) >= 2:
                        if order == 'C':
                            slicing = slice(-2, None)
                        else:
                            slicing = slice(None, 2)
                        axes = tuple(resample_axes[slicing])
                    else:
                        if order == 'C':
                            axes = (resample_axes[0], axes_order[-1][0])
                        else:
                            axes = (axes_order[-1][1], resample_axes[0])

                    axes_order.append(axes)
                    last_shape = tuple([s if d not in axes else t for d, s, t in
                                        zip(range(ndim), last_shape, sink_shape_in_source_orientation)])
                    shape_order.append(last_shape)
                    resample_axes = np.array([a for a in resample_axes if a not in axes])

                return axes_order, shape_order

        else:
            raise ValueError("axes_order %r not 'size','order' or list but %r!" % axes_order)


def resample(source, sink=None, orientation=None,
             sink_shape=None, source_resolution=None, sink_resolution=None,
             interpolation='linear', axes_order=None, method='shared',
             processes=None, workspace=None, verbose=True):
    """Resample data of source in new shape/resolution and orientation.

    Arguments
    ---------
    source : str or array
      The source to be resampled.
    sink : str or None
      The sink for the resampled image.
    orientation : tuple or None:
      The orientation specified by permuation and change in sign of (1,2,3).
    sink_shape : tuple or None
      The target shape of the resampled sink.
    source_resolution : tuple or None
      The resolution of the source (in length per pixel).
    sink_resolution : tuple or None
      The resolution of the resampled source (in length per pixel).
    interpolation : str
      The method to use for interpolating to the resmapled array.
    axis_order : str, list of tuples of int or None
      The axes pairs along which to resample the data at each step.
      If None, this is detertmined automatically. For a FileList source,
      setting the first tuple should point to axis not indicating files.
      If 'size' the axis order is determined automatically to maximally reduce
      the size of the array in each resmapling step.
      If 'order' the axis order is chosed automatically to optimize io speed.
    method : 'shared' or 'memmap'
      Method to handle intermediate resampling results. If 'shared' use shared
      memory, otherwise use a memory map on disk.
    processes : int, None or 'serial'
      Number of processes to use for parallel resampling, if None use maximal
      processes avaialable, if 'serial' process in serial.
    verbose : bool
      If True, display progress information.

    Returns
    -------
    sink : array or str
      The data or filename of resampled sink.

    Notes
    -----
    * Resolutions are assumed to be given for the axes of the intrinsic
      orientation of the data and reference (as when viewed by ImageJ).
    * Orientation: permuation of 1,2,3 with potential sign, indicating which
      axes map onto the reference axes, a negative sign indicates reversal
      of that particular axes.
    * Only a minimal set of information to determine the resampling parameter
      has to be given, e.g. source_shape and sink_shape.
    * The resampling is done by iterating two dimensional resampling steps.
    """

    source = io.as_source(source)
    source_shape = source.shape
    ndim = len(source_shape)
    dtype = source.dtype
    order = source.order

    orientation = format_orientation(orientation)

    source_shape, sink_shape, source_resolution, sink_resolution = \
        resample_shape(source_shape=source_shape, sink_shape=sink_shape,
                       source_resolution=source_resolution, sink_resolution=sink_resolution,
                       orientation=orientation)

    sink_shape_in_source_orientation = orient_shape(sink_shape, orientation, inverse=True)

    interpolation = _interpolation_to_cv2(interpolation)

    if not isinstance(processes, int) and processes != 'serial':
        processes = mp.cpu_count()

    # detemine order of resampling
    axes_order, shape_order = _axes_order(axes_order, source, sink_shape_in_source_orientation, order=order)
    # print(axes_order, shape_order)

    if len(axes_order) == 0:
        if verbose:
            print('resampling: no resampling necessary, source has same size as sink!')
        if sink != source:
            return io.write(sink, source)
        else:
            return source

    # resample
    n_steps = len(axes_order)
    last_source = source
    delete_files = []
    for step, axes, shape in zip(range(n_steps), axes_order, shape_order):
        if step == n_steps - 1 and orientation is None:
            resampled = io.initialize(source=sink, shape=sink_shape, dtype=dtype, as_source=True)
        else:
            if method == 'shared':
                resampled = io.sma.create(shape, dtype=dtype, order=order, as_source=True)
            else:
                location = tempfile.mktemp() + '.npy'
                resampled = io.mmp.create(location, shape=shape, dtype=dtype, order=order, as_source=True)
                delete_files.append(location)
        # print(resampled)

        # indices for non-resampled axes
        indices = tuple([range(s) for d, s in enumerate(shape) if d not in axes])
        indices = [i for i in itertools.product(*indices)]
        n_indices = len(indices)

        # resample step
        last_source_virtual = last_source.as_virtual()
        resampled_virtual = resampled.as_virtual()
        _resample = ft.partial(_resample_2d, source=last_source_virtual, sink=resampled_virtual, axes=axes, shape=shape,
                               interpolation=interpolation, n_indices=n_indices, verbose=verbose)

        if processes == 'serial':
            for index in indices:
                _resample(index=index)
        else:
            with ThreadPoolExecutor(processes) as executor:
                chunk_size = round(len(indices) / (processes * 3))
                executor.map(_resample, indices, chunksize=chunk_size)
                if workspace is not None:
                    workspace.executor = executor
        last_source = resampled

    # fix orientation
    if orientation is not None:
        # permute
        per = orientation_to_permuation(orientation)
        resampled = resampled.transpose(per)

        # reverse axes
        reslice = False
        slicing = [slice(None)] * ndim
        for d, o in enumerate(orientation):
            if o < 0:
                slicing[d] = slice(None, None, -1)
                reslice = True
        if reslice:
            resampled = resampled[slicing]

        if verbose:
            print("resample: re-oriented shape %r!" % (resampled.shape,))

        sink = io.write(sink, resampled)
    else:
        sink = resampled

    for f in delete_files:
        fu.delete_file(f)

    return sink


def resample_files(sample_name, sample_directory, overwrite=False, **kwargs):
    print("")
    for channel in kwargs["study_params"]["channels_to_stitch"]:
        converted_file = os.path.join(sample_directory, f"stitched_{channel}.npy")
        resampled_25_path = os.path.join(sample_directory, f"resampled_25um_{channel}.tif")
        resampled_10_path = os.path.join(sample_directory, f"resampled_10um_{channel}.tif")

        # Fetch metadata for the sample being processed
        metadata_file = os.path.join(sample_directory, "scan_metadata.json")
        if not os.path.exists(metadata_file):
            raise ut.CmliteError("Metadata file is missing!")
        else:
            with open(metadata_file, 'r') as json_file:
                metadata = json.load(json_file)
        resolution = np.array([metadata["x_res"], metadata["y_res"], metadata["z_res"]])

        if not os.path.exists(resampled_25_path) or overwrite:
            ut.print_c(f"[INFO {sample_name}] Resampling channel {channel} to 25um isotropic!")
            resample_25um_parameter = {
                "source_resolution": resolution,
                "sink_resolution": (25, 25, 25),
                "processes": None,
                "verbose": True,
                "method": "memmap",
            }
            fu.delete_file(resampled_25_path)
            resample(converted_file, sink=resampled_25_path, **resample_25um_parameter)
        else:
            ut.print_c(f"[WARNING {sample_name}] Resampling (25um): skipped for channel {channel}: "
                       f"resampled_25um_{channel}.tif file already exists!")

        if not os.path.exists(resampled_10_path) or overwrite:
            if channel != kwargs["study_params"]["autofluorescence_channel"]:
                ut.print_c(f"[INFO {sample_name}] Resampling channel {channel} to 10um isotropic!")
                resample_10um_parameter = {
                    "source_resolution": resolution,
                    "sink_resolution": (10, 10, 10),
                    "processes": None,
                    "verbose": True,
                    "method": "memmap",
                }
                resampled_10_path = os.path.join(sample_directory, f"resampled_10um_{channel}.tif")
                fu.delete_file(resampled_10_path)
                resample(converted_file, sink=resampled_10_path, **resample_10um_parameter)
        else:
            ut.print_c(f"[WARNING {sample_name}] Resampling (10um): skipped for channel {channel}: "
                       f"resampled_25um_{channel}.tif file already exists!")

def resample_points(source, sink=None, resample_source=None, resample_sink=None,
                    orientation=None, source_shape=None, sink_shape=None,
                    source_resolution=None, sink_resolution=None, **args):
    """Resample points from original coordiantes to resampled ones.

    Arguments
    ---------
    source : str or array
      Points to be resampled.
    sink : str or None
      Sink for the resampled point coordinates.
    orientation : tuple
      Orientation as specified in :func:`resample`.
    resample_source : str, array or None
      Optional source as in :func:`resample`.
    resample_sink: str, array or None
      Optional sink used in :func:`resample`.
    source_shape : tuple or None
      Optional value of source_shape as in :func:`resample`.
    source_resolution : tuple or None
      Optional value of source_resolution as in :func:`resample`.
    sink_resolution : tuple or None
      Optional value of sink_resolution as in :func:`resample`.

    Returns
    -------
    resmapled : array or str
      Sink for the resampled point coordinates.

    Notes
    -----
    * The resampling of points here corresponds to he resampling of an image
      in :func:`resample`.
    * The arguments should be passed exactly as in :func:`resample` except soure
      and sink that point to the point sources.
      Use resample_source and resmaple_sink to pass the source and sink values
      used in :func:`resample`.
    """
    # orientation
    orientation = format_orientation(orientation)

    # original source info
    if source_shape is None:
        if source_resolution is None and resample_source is None:
            raise ValueError('Either source_shape, source_resolution or resample_source must to be given!')
        if resample_source is not None:
            source_shape = io.shape(resample_source)

    # original sink info
    if sink_shape is None and sink_resolution is None:
        if resample_sink is None:
            sink_shape = io.shape(source)
        else:
            sink_shape = io.shape(resample_sink)

    source_shape, sink_shape, source_resolution, sink_resolution = \
        resample_shape(source_shape=source_shape, sink_shape=sink_shape,
                       source_resolution=source_resolution, sink_resolution=sink_resolution,
                       orientation=orientation)

    sink_shape_in_source_orientation = orient_shape(sink_shape, orientation, inverse=True)

    resample_factor = [float(t) / float(s) for s, t in zip(source_shape, sink_shape_in_source_orientation)]

    points = io.as_source(source)
    resampled = points[:] * resample_factor

    # reorient points
    if orientation is not None:
        # permute
        per = orientation_to_permuation(orientation)
        resampled = resampled.transpose(per)

        # reverse axes
        reslice = False
        slicing = [slice(None)] * len(source_shape)
        for d, o in enumerate(orientation):
            if o < 0:
                slicing[d] = slice(None, None, -1)
                reslice = True
        if reslice:
            resampled = resampled[slicing]

    return io.write(sink, resampled)
import math
import numpy as np

import pyximport

pyximport.install(setup_args={"include_dirs": np.get_include()}, reload_support=True)

import IO.IO as io

import parallel_processing.data_processing.array_processing as ap

import parallel_processing.data_processing.devolve_point_list_code as code


###############################################################################
### Voxelization
###############################################################################

def devolve(source, sink=None, shape=None, dtype=None,
            weights=None, intensity=False, indices=None, kernel=None,
            processes=None, verbose=False):
    """Converts a list of points into an volumetric image array.
  
  Arguments
  ---------
  source : str, array or Source
    Source of point of nxd coordinates.
  sink : str, array or None
    The sink for the devolved image, if None return array.
  shape : tuple, str or None
    Shape of the final devolved data. If None, determine from points.
    If str, determine shape from the source at the specified location.
  dtype : dtype or None
    Optional data type of the sink.
  weights : array or None
    Weight array of length n for each point. If None, use uniform weights.  
  method : str
    Method for voxelization: 'sphere', 'rectangle' or 'pixel'.
  indices : array 
    The relative indices to the center to devolve over as nxd array.
  kernel : array
    Optional kernel weights for each index in indices.
  processes : int or None
    Number of processes to use.
  verbose : bool
    If True, print progress info.                        
 
  Returns
  -------
  sink : str, array
    Volumetric data of devolved point data.
  """
    processes, timer = ap.initialize_processing(processes=processes, verbose=verbose, function='devolve')

    # points, points_buffer = ap.initialize_source(points)
    points_buffer = io.as_source(source).as_buffer()
    if points_buffer.ndim == 1:
        points_buffer = points_buffer[:, None]

    if sink is None and shape is None:
        if points_buffer.ndim > 1:
            shape = tuple(int(math.ceil(points_buffer[:, d].max())) for d in range(points_buffer.shape[1]))
        else:
            shape = (int(math.ceil(points_buffer[:].max())),)
    elif isinstance(shape, str):
        shape = io.shape(shape)

    if sink is None and dtype is None:
        if weights is not None:
            dtype = io.dtype(weights)
        elif kernel is not None:
            kernel = np.asarray(kernel)
            dtype = kernel.dtype
        else:
            dtype = int

    sink, sink_buffer, sink_shape, sink_strides, = ap.initialize_sink(sink=sink, shape=shape, dtype=dtype,
                                                                      return_shape=True, return_strides=True,
                                                                      as_1d=True)

    if indices is None:
        return sink
    indices = np.asarray(indices, dtype=int)
    if indices.ndim == 1:
        indices = indices[:, None]

    if kernel is not None:
        kernel = np.asarray(kernel, dtype=float)

    points_buffer = points_buffer.astype(np.float64)
    indices = indices.astype(np.intp)
    sink_shape = sink_shape.astype(np.intp)
    sink_strides = sink_strides.astype(np.intp)
    if weights is not None:
        weights = weights.astype(np.float64)

    if weights is None:
        if kernel is None:
            code.devolve_uniform(points_buffer, indices, sink_buffer, sink_shape, sink_strides, processes)
        else:
            code.devolve_uniform_kernel(points_buffer, indices, kernel, sink_buffer, sink_shape, sink_strides,
                                        processes)
    else:
        if kernel is None:
            if intensity:
                # Initialize the weight_sum array
                weight_sum = np.zeros(sink_shape, dtype=sink_buffer.dtype)

                # Call devolve_weights_intensity with the weight_sum array
                code.devolve_weights_intensity(points_buffer, weights, indices, sink_buffer, weight_sum.ravel(),
                                               sink_shape, sink_strides, processes)

                # Reshape sink_buffer and weight_sum to their original shape
                sink_buffer = sink_buffer.reshape(sink_shape)
                weight_sum = weight_sum.reshape(sink_shape)

                # Normalize the sink_buffer to get the average
                with np.errstate(divide='ignore', invalid='ignore'):
                    sink = np.divide(sink_buffer, weight_sum, out=np.zeros_like(sink_buffer),
                                     where=weight_sum != 0)
            else:
                code.devolve_weights(points_buffer, weights, indices, sink_buffer, sink_shape, sink_strides, processes)
        else:
            code.devolve_weights_kernel(points_buffer, weights, indices, kernel, sink_buffer, sink_shape, sink_strides,
                                        processes)

    ap.finalize_processing(verbose=verbose, function='devolve', timer=timer)

    return sink

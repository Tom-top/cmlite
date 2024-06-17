import numpy as np

import IO.IO as io

import parallel_processing.data_processing.measure_point_list as mpl

import utils.timer as tmr


###############################################################################
### Measure Radius
###############################################################################

def measure_expression(source, points, search_radius, method='max',
                       sink=None, processes=None, verbose=False):
    """Measures the expression around a list of points in a source.

    Arguments
    ---------
    source : array
      Source for measurement.
    points : array
      List of indices to measure radis for.
    search_radius : int or array
      List of search radii to use around each point. If int  use
      this radius for all points. Array should be of length of points.
    method : 'max' or 'min', 'mean'
      Measurement type.
    processes : int or None
      Number of processes to use.
    verbose : bool
      If True, print progress info.


    """
    source = io.as_source(source)
    ndim = source.ndim

    if verbose:
        timer = tmr.Timer()
        print('Measuring expression of %d points in array of shape %r.' % (points.shape[0], source.shape))

    if not hasattr(search_radius, '__len__'):
        search_radius = search_radius * np.ones(points.shape[0])
    if len(search_radius) != len(points):
        raise ValueError('The search_radius is not valid!')

    indices, radii_indices = search_indices(search_radius, ndim)

    if method == 'max':
        expression = mpl.measure_max(source, points, indices, radii_indices, sink=sink, processes=processes,
                                     verbose=verbose)
    elif method == 'min':
        expression = mpl.measure_min(source, points, indices, radii_indices, sink=sink, processes=processes,
                                     verbose=verbose)
    elif method == 'mean':
        expression = mpl.measure_mean(source, points, indices, radii_indices, sink=sink, processes=processes,
                                      verbose=verbose)
    elif method == 'sum':
        expression = mpl.measure_sum(source, points, indices, radii_indices, sink=sink, processes=processes,
                                     verbose=verbose)
    else:
        raise ValueError("Method %r not in 'max', 'min', 'mean'" % method)

    if verbose:
        timer.print_elapsed_time('Measuring expression done')

    return expression


###############################################################################
### Search indices
###############################################################################

def search_indices(radii, ndim):
    """Creates all relative indices within a sphere of specified radius in an array with specified strides.

    Arguments
    ---------
    radius : tuple or float
      Radius of the sphere of the search index list.
    strides : tuple of ints
      Srides of the array
    scale : float, tuple or None
      Spatial scale in each array dimension.

    Returns
    -------
    indices : array
       Array of ints of relative indices for the search area voxels.
    """
    radius = int(np.ceil(np.max(radii)))

    # create coordiante grid
    grid = [np.arange(-radius, radius + 1)] * ndim
    grid = np.array(np.meshgrid(*grid, indexing='ij'))

    # sort indices by radius
    dist = np.sum(grid * grid, axis=0)
    dist_shape = dist.shape
    dist = dist.reshape(-1)
    dist_index = np.argsort(dist)
    dist = np.sqrt(dist[dist_index])
    dist = np.hstack([dist, np.inf])
    # print(dist)

    radii_indices = np.searchsorted(dist, radii, side='right')

    # convert coordinates to full indices via strides
    indices = np.array(np.unravel_index(dist_index, dist_shape)).T
    indices -= radius

    # remove center point
    indices = indices[1:]
    radii_indices[radii_indices > 0] -= 1

    return indices, radii_indices
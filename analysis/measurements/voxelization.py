import os
import shutil

import numpy as np

import IO.IO as io

import parallel_processing.data_processing.devolve_point_list as dpl


###############################################################################
### Voxelization
###############################################################################

def voxelize(source, sink=None, shape=None, dtype=None, weights=None,
             method='sphere', radius=(1, 1, 1), kernel=None,
             processes=None, verbose=False):
    """Converts a list of points into an volumetric image array

    Arguments
    ---------
    source : str, array or Source
      Source of point of nxd coordinates.
    sink : str, array or None
      The sink for the voxelized image, if None return array.
    shape : tuple or None
      Shape of the final voxelized data. If None, deterimine from points.
    dtype : dtype or None
      Optional data type of the sink.
    weights : array or None
      Weight array of length n for each point. If None, use uniform weights.
    method : str
      Method for voxelization: 'sphere', 'rectangle' or 'pixel'.
    radius : tuple
      Radius of the voxel region to integrate over.
    kernel : function
      Optional function of distance to set weights in the voxelization.
    processes : int or None
      Number of processes to use.
    verbose : bool
      If True, print progress info.

    Returns
    -------
    sink : str, array
      Volumetric data of voxelied point data.
    """
    points = io.read(source)

    points_shape = points.shape
    if len(points_shape) > 1:
        ndim = points_shape[1]
    else:
        ndim = 1

    if not hasattr(radius, '__len__'):
        radius = [radius] * ndim
    if len(radius) != ndim:
        raise ValueError('Radius %r and points with shape %r do not match in dimension!' % (radius, points_shape))

    if method == 'sphere':
        indices, kernel = search_indices_sphere(radius, kernel)
    elif method == 'rectangle':
        indices, kernel = search_indices_rectangle(radius, kernel)
    elif method == 'pixel':
        indices = np.array(0, dtype=int)
        if kernel is not None:
            kernel = np.array([kernel(0)])
    else:
        raise ValueError("method not 'sphere', 'rectangle', or 'pixel', but %r!" % method)

    return dpl.devolve(points, sink=sink, shape=shape, dtype=dtype,
                       weights=weights, indices=indices, kernel=kernel, processes=processes, verbose=verbose)


###############################################################################
### Search indices
###############################################################################

def search_indices_sphere(radius, kernel=None):
    """Creates all relative indices within a sphere of specified radius.

    Arguments
    ---------
    radius : tuple or int
      Radius of the sphere of the search index list.

    Returns
    -------
    indices : array
       Array of ints of relative indices for the search area voxels.
    """
    # create coordiante grid
    grid = [np.arange(-r, r + 1, dtype=float) / np.maximum(1, r) for r in radius]
    grid = np.array(np.meshgrid(*grid, indexing='ij'))

    # sort indices by radius
    dist = np.sum(grid * grid, axis=0)
    dist_shape = dist.shape
    dist = dist.reshape(-1)
    dist_index = np.argsort(dist)
    dist_sorted = dist[dist_index]
    keep = dist_sorted <= 1
    dist_index = dist_index[keep]

    if kernel is not None:
        dist_sorted = np.sqrt(dist_sorted[keep])
        kernel = np.array([kernel(d) for d in dist_sorted], dtype=float)
    else:
        kernel = None

    # convert to relative coordinates
    indices = np.array(np.unravel_index(dist_index, dist_shape)).T
    indices -= radius

    return indices, kernel


def search_indices_rectangle(radius, kernel=None):
    """Creates all relative indices within a rectangle.

    Arguments
    ---------
    radius : tuple or float
      Radius of the sphere of the search index list.

    Returns
    -------
    indices : array
       Array of ints of relative indices for the search area voxels.
    """
    # create coordiante grid
    grid = [np.arange(-r, r + 1, dtype=int) for r in radius]
    grid = np.array(np.meshgrid(*grid, indexing='ij'))

    if kernel is not None:
        dist = np.sqrt(np.sum(grid * grid, axis=0))
        kernel = np.array([kernel(d) for d in dist], dtype=float)
    else:
        kernel = None

    indices = grid.reshape((len(radius), -1)).T

    return indices, kernel


def generate_heatmap(sample_name, sample_directory, analysis_data_size_directory, annotation_files, weighed=False,
                     **kwargs):
    shape_detection_directory = os.path.join(sample_directory,
                                             f"shape_detection_{kwargs['cell_detection']['shape_detection']}")
    if not os.path.exists(shape_detection_directory):
        os.mkdir(shape_detection_directory)
    for channel in kwargs['study_params']['channels_to_segment']:
        cd_p = kwargs["cell_detection"]
        vox_p = kwargs["voxelization"]
        for atlas_name, annotation_file in zip(kwargs["study_params"]["atlas_to_use"], annotation_files):
            shape_detection_directory = os.path.join(sample_directory, f"shape_detection_{cd_p['shape_detection']}")
            cells_transformed_path = os.path.join(shape_detection_directory, f"{atlas_name}_cells_transformed_"
                                                                             f"{channel}.npy")
            cells_transformed = io.as_source(cells_transformed_path)
            coordinates = np.array([cells_transformed[n] for n in ['xt', 'yt', 'zt']]).T

            if weighed:
                weights = cells_transformed['source']
            else:
                weights = None

            annotation_shape = io.shape(annotation_file)
            voxelization_parameter = dict(
                shape=annotation_shape,
                dtype=None,
                weights=weights,
                method='sphere',
                radius=vox_p["radius"],
                kernel=None,
                processes=None,
                verbose=True,
            )

            if weighed:
                heatmap_name = f"{atlas_name}_density_intensities_{channel}.tif"
            else:
                heatmap_name = f"{atlas_name}_density_counts_{channel}.tif"

            heatmap_path = os.path.join(shape_detection_directory, heatmap_name)
            if os.path.exists(heatmap_path):
                os.remove(heatmap_path)
            analysis_sample_directory = os.path.join(analysis_data_size_directory, sample_name)

            voxelize(coordinates, sink=heatmap_path, **voxelization_parameter)
            shutil.copyfile(heatmap_path,
                            os.path.join(analysis_sample_directory, heatmap_name))
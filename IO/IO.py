import os

import numpy as np
from natsort import natsorted
import importlib
import h5py

import IO.source as src
import IO.slice as slc
import IO.TIF as tif
import IO.NRRD as nrrd
import IO.CSV as csv
import IO.NPY as npy
import IO.MMP as mmp
import IO.SMA as sma
import IO.file_list as fl
import IO.file_utils as fu

import utils.utils as ut

import utils.tag_expression as te

data_file_extensions = ["tif", "tiff", "mhd", "nrrd", "npy"]
"""list of extensions supported as a iswmage data file"""

data_file_extensions_to_type = {"npy": "MMP", "tif": "TIF", "tiff": "TIF", "mhd": "RAW", "nrrd": "NRRD"}
"""map from image file extensions to image file types"""

source_modules = [npy, tif, mmp, sma, fl, nrrd, csv]
"""The valid source modules."""

file_extension_to_module = {"npy": mmp, "tif": tif, "tiff": tif, 'nrrd': nrrd,
                            'nrdh': nrrd, 'csv': csv}
"""Map between file extensions and modules that handle this file type."""


def source_to_module(source):
    """Returns IO module associated with a source.

    Arguments
    ---------
    source : object
      The source specification.

    Returns
    -------
    type : module
      The module that handles the IO of the source.
    """
    if isinstance(source, src.Source):
        return importlib.import_module(source.__module__)
    elif isinstance(source, (str, te.Expression)):
        return location_to_module(source)
    elif isinstance(source, np.memmap):
        return mmp
    elif isinstance(source, (np.ndarray, list, tuple)) or source is None:
        if sma.is_shared(source):
            return sma
        else:
            return npy
    else:
        raise ValueError('The source %r is not a valid source!' % source)


def location_to_module(location):
    """Returns the IO module associated with a location string.

    Arguments
    ---------
    location : object
      Location of the source.

    Returns
    -------
    module : module
      The module that handles the IO of the source specified by its location.
    """
    if fl.is_file_list(location):
        return fl
    else:
        return filename_to_module(location)


def filename_to_module(filename):
    """Returns the IO module associated with a filename.

    Arguments
    ---------
    filename : str
      The file name.

    Returns
    -------
    module : module
      The module that handles the IO of the file.
    """
    ext = fu.file_extension(filename)

    mod = file_extension_to_module.get(ext, None)
    if mod is None:
        raise ValueError("Cannot determine module for file %s with extension %s!" % (filename, ext))

    return mod


def is_source(source, exists=True):
    """Checks if source is a valid source.

    Arguments
    ---------
    source : object
      Source to check.
    exists : bool
      If True, check if source exists in case it has a location.

    Returns
    -------
    is_source : bool
      True if source is a valid source.
    """
    if isinstance(source, src.Source):
        if exists:
            return source.exists()
        else:
            return True

    elif isinstance(source, str):
        try:
            mod = location_to_module(source)
        except:
            return False
        if exists:
            return mod.Source(source).exists()
        else:
            return True

    elif isinstance(source, np.memmap):
        return True

    elif isinstance(source, (np.ndarray, list, tuple)):
        return True

    else:
        return False


def as_source(source, slicing=None, *args, **kwargs):
    """Convert source specification to a Source class.

    Arguments
    ---------
    source : object
      The source specification.

    Returns
    -------
    source : Source class
      The source class.
    """
    if not isinstance(source, src.Source):
        mod = source_to_module(source)
        source = mod.Source(source, *args, **kwargs)
    if slicing is not None:
        source = slc.Slice(source=source, slicing=slicing)
    return source


def source(source, slicing=None, *args, **kwargs):
    """Convert source specification to a Source class.

    Arguments
    ---------
    source : object
      The source specification.

    Returns
    -------
    source : Source class
      The source class.
    """
    return as_source(source, slicing=slicing, *args, **kwargs)


def ndim(source):
    """Returns number of dimensions of a source.

    Arguments
    ---------
    source : str, array or Source
      The source specification.

    Returns
    -------
    ndim : int
      The number of dimensions in the source.
    """
    source = as_source(source)
    return source.ndim


def shape(source):
    """Returns shape of a source.

    Arguments
    ---------
    source : str, array or Source
      The source specification.

    Returns
    -------
    shape : tuple of ints
      The shape of the source.
    """
    source = as_source(source)
    return source.shape


def size(source):
    """Returns size of a source.

    Arguments
    ---------
    source : str, array or Source
      The source specification.

    Returns
    -------
    size : int
      The size of the source.
    """
    source = as_source(source)
    return source.size


def dtype(source):
    """Returns dtype of a source.

    Arguments
    ---------
    source : str, array or Source
      The source specification.

    Returns
    -------
    dtype : dtype
      The data type of the source.
    """
    source = as_source(source)
    return source.dtype


def order(source):
    """Returns order of a source.

    Arguments
    ---------
    source : str, array or Source
      The source specification.

    Returns
    -------
    order : 'C', 'F', or None
      The order of the source data items.
    """
    source = as_source(source)
    return source.order


def location(source):
    """Returns the location of a source.

    Arguments
    ---------
    source : str, array or Source
      The source specification.

    Returns
    -------
    location : str or None
      The location of the source.
    """
    source = as_source(source)
    return source.location


def memory(source):
    """Returns the memory type of a source.

    Arguments
    ---------
    source : str, array or Source
      The source specification.

    Returns
    -------
    memory : str or None
      The memory type of the source.
    """
    if sma.is_shared(source):
        return 'shared'
    else:
        return None


def element_strides(source):
    """Returns the strides of the data array of a source.

    Arguments
    ---------
    source : str, array, dtype or Source
      The source specification.

    Returns
    -------
    strides : tuple of int
      The strides of the souce.
    """
    try:
        source = as_source(source)
        strides = source.element_strides
    except:
        raise ValueError('Cannot determine the strides for the source!')

    return strides


def buffer(source):
    """Returns an io buffer of the data array of a source for use with e,g python.

    Arguments
    ---------
    source : source specification
      The source specification.

    Returns
    -------
    buffer : array or memmap
      A buffer to read and write data.
    """
    try:
        source = as_source(source)
        buffer = source.as_buffer()
    except:
        raise ValueError('Cannot get a io buffer for the source!')

    return buffer


# TODO: arg memory= to specify which kind of array is created, better use device=
# TODO: arg processes= in order to use ParallelIO -> can combine with buffer=
def read(source, *args, **kwargs):
    """Read data from a data source.

    Arguments
    ---------
    source : str, array, Source class
      The source to read the data from.

    Returns
    -------
    data : array
      The data of the source.
    """
    mod = source_to_module(source)
    return mod.read(source, *args, **kwargs)


def write(sink, data, *args, **kwargs):
    """Write data to a data source.

    Arguments
    ---------
    sink : str, array, Source class
      The source to write data to.
    data : array
      The data to write to the sink.
    slicing : slice specification or None
      Optional subslice to write data to.

    Returns
    -------
    sink : str, array or Source class
      The sink to which the data was written.
    """
    mod = source_to_module(sink)
    return mod.write(sink, as_source(data), *args, **kwargs)


def create(source, *args, **kwargs):
    """Create a data source on disk.

    Arguments
    ---------
    source : str, array, Source class
      The source to write data to.

    Returns
    -------
    sink : str, array or Source class
      The sink to which the data was written.
    """
    mod = source_to_module(source)
    return mod.create(source, *args, **kwargs)


def initialize(source=None, shape=None, dtype=None, order=None, location=None, memory=None, like=None, hint=None,
               **kwargs):
    """Initialize a source with specified properties.

    Arguments
    ---------
    source : str, array, Source class
      The source to write data to.
    shape : tuple or None
      The desired shape of the source.
      If None, infered from existing file or from the like parameter.
      If not None and source has a valid shape shapes are tested to match.
    dtype : type, str or None
      The desired dtype of the source.
      If None, infered from existing file or from the like parameter.
      If not None and source has a valid dtype the types are tested to match.
    order : 'C', 'F' or None
      The desired order of the source.
      If None, infered from existing file or from the like parameter.
      If not None and source has a valid order the orders are tested to match.
    location : str or None
      The desired location of the source.
      If None, infered from existing file or from the like parameter.
      If not None and source has a valid location the locations need to match.
    memory : 'shared' or None
      The memory type of the source. If 'shared' a shared array is created.
    like : str, array or Source class
      Infer the source parameter from this source.
    hint : str, array or Source class
      If parameters for source creation are missing use the ones from this
      hint source.

    Returns
    -------
    source : Source class
      The initialized source.

    Note
    ----
    The source is created on disk or in memory if it does not exists so processes
    can start writing into it.
    """
    if isinstance(source, (str, te.Expression)):
        location = source
        source = None

    if like is not None:
        like = as_source(like)
        if shape is None:
            shape = like.shape
        if dtype is None:
            dtype = like.dtype
        if order is None:
            order = like.order

    if source is None:
        if location is None:
            shape, dtype, order = _from_hint(hint, shape, dtype, order)
            if memory == 'shared':
                return sma.create(shape=shape, dtype=dtype, order=order, **kwargs)
            else:
                return npy.create(shape=shape, dtype=dtype, order=order)
        else:
            try:
                source = as_source(location)
            except:
                try:
                    shape, dtype, order = _from_hint(hint, shape, dtype, order)
                    mod = location_to_module(location)
                    return mod.create(location=location, shape=shape, dtype=dtype, order=order,
                                      **kwargs)
                except Exception as error:
                    raise ValueError(f'Cannot initialize source for location {location} - {error}')

    if isinstance(source, np.ndarray):
        source = as_source(source)

    if not isinstance(source, src.Source):
        raise ValueError('Source specification %r not a valid location, array or Source class!' % source)

    if shape is not None and shape != source.shape:
        raise ValueError('Incompatible shapes %r != %r for the source %r!' % (shape, source.shape, source))
    if dtype is not None and dtype != source.dtype:
        raise ValueError('Incompatible dtype %r != %r for the source %r!' % (dtype, source.dtype, source))
    if order is not None and order != source.order:
        raise ValueError('Incompatible order %r != %r for the source %r!' % (order, source.order, source))
    if location is not None and fu.abspath(location) != fu.abspath(source.location):
        raise ValueError('Incompatible location %r != %r for the source %r!' % (location, source.location, source))
    if memory == 'shared' and not sma.is_shared(source):
        raise ValueError('Incompatible memory type, the source %r is not shared!' % (source,))

    return source


def _from_hint(hint, shape, dtype, order):
    """Helper for initialize."""
    if hint is not None:
        try:
            hint = as_source(hint)
            if shape is None:
                shape = hint.shape
            if dtype is None:
                dtype = hint.dtype
            if order is None:
                order = hint.order
        except:
            pass
    return shape, dtype, order


def initialize_buffer(source, shape=None, dtype=None, order=None, location=None, memory=None, like=None, **kwargs):
    """Initialize a buffer with specific properties.

    Arguments
    ---------
    source : str, array, Source class
      The source to write data to.
    shape : tuple or None
      The desired shape of the source.
      If None, infered from existing file or from the like parameter.
      If not None and source has a valid shape shapes are tested to match.
    dtype : type, str or None
      The desired dtype of the source.
      If None, infered from existing file or from the like parameter.
      If not None and source has a valid dtype the types are tested to match.
    order : 'C', 'F' or None
      The desired order of the source.
      If None, infered from existing file or from the like parameter.
      If not None and source has a valid order the orders are tested to match.
    location : str or None
      The desired location of the source.
      If None, infered from existing file or from the like parameter.
      If not None and source has a valid location the locations need to match.
    memory : 'shared' or None
      The memory type of the source. If 'shared' a shared array is created.
    like : str, array or Source class
      Infer the source parameter from this source.

    Returns
    -------
    buffer : array
      The initialized buffer to use tih e.g. cython.

    Note
    ----
    The buffer is created if it does not exists.
    """
    source = initialize(source, shape=shape, dtype=dtype, order=order, location=location, memory=memory, **kwargs)
    return source.as_buffer()


###############################################################################
### Utils
###############################################################################

def file_list(expression=None, file_list=None, sort=True, verbose=False):
    """Returns the list of files that match the tag expression.

    Arguments
    ---------
    expression :str
      The regular expression the file names should match.
    sort : bool
      If True, sort files naturally.
    verbose : bool
      If True, print warning if no files exists.

    Returns
    -------
    file_list : list of str
      The list of files that matched the expression.
    """
    return fl._file_list(expression=expression, file_list=file_list, sort=sort, verbose=verbose)


def max_value(source):
    """Returns the maximal value of the data type of a source.

    Arguments
    ---------
    source : str, array, dtype or Source
      The source specification.

    Returns
    -------
    max_value : number
      The maximal value for the data type of the source
    """
    if isinstance(source, (src.Source, np.ndarray)):
        source = source.dtype

    if isinstance(source, str):
        try:
            source = np.dtype(source)
        except:
            pass

    if not isinstance(source, (type, np.dtype)):
        source = dtype(source)

    try:
        max_value = np.iinfo(source).max
    except:
        try:
            max_value = np.finfo(source).max
        except:
            raise ValueError('Cannot determine the maximal value for the type %r!' % source)
    return max_value


def min_value(source):
    """Returns the minimal value of the data type of a source.

    Arguments
    ---------
    source : str, array, dtype or Source
      The source specification.

    Returns
    -------
    min_value : number
      The minimal value for the data type of the source
    """
    if isinstance(source, str):
        try:
            source = np.dtype(source)
        except:
            pass

    if not isinstance(source, (type, np.dtype)):
        source = dtype(source)

    try:
        min_value = np.iinfo(source).min
    except:
        try:
            min_value = np.finfo(source).min
        except:
            raise ValueError('Cannot determine the minimal value for the type %r!' % source)
    return min_value


def data_filename_to_module(filename):
    """Return the module that handles IO for a data file

    Arguments:
        filename (str): file name

    Returns:
        object: sub-module that handles a specific data type
    """

    ft = data_filename_to_type(filename)
    return importlib.import_module("IO." + ft)


def data_filename_to_type(filename):
    """Returns type of a image data file

    Arguments:
        filename (str): file name

    Returns:
        str: image data type in :const:`dataFileTypes`
    """

    if fu.is_file_expression(filename):
        return "FileList"
    else:
        fext = fu.file_extension(filename)
        if fext in data_file_extensions:
            return data_file_extensions_to_type[fext]
        else:
            raise RuntimeError("Cannot determine type of data file %s with extension %s" % (filename, fext))


def data_size(source, x=all, y=all, z=all, **args):
    """Returns array size of the image data needed when read from file and reduced to specified ranges

    Arguments:
        source (array or str): source data
        x,y,z (tuple or all): range specifications, ``all`` is full range

    Returns:
        tuple: size of the image data after reading and range reduction
    """

    if isinstance(source, str):
        mod = data_filename_to_module(source)
        return mod.data_size(source, x=x, y=y, z=z, **args)
    elif isinstance(source, np.ndarray):
        return data_size_from_data_range(source.shape, x=x, y=y, z=z)
    elif isinstance(source, tuple):
        return data_size_from_data_range(source, x=x, y=y, z=z)
    else:
        raise RuntimeError("dataSize: argument not a string, tuple or array!")


def data_size_from_data_range(data_size, x=all, y=all, z=all, **args):
    """Converts full data size to actual size given ranges for x,y,z

    Arguments:
        data_size (tuple): data size
        x,z,y (tuple or all): range specifications, ``all`` is full range

    Returns:
        tuple: data size as tuple of integers

    See Also:
        :func:`toDataRange`, :func:`to_data_size`
    """

    data_size = list(data_size)
    n = len(data_size)
    if n > 0:
        data_size[0] = to_data_size(data_size[0], r=x)
    if n > 1:
        data_size[1] = to_data_size(data_size[1], r=y)
    if n > 2:
        data_size[2] = to_data_size(data_size[2], r=z)

    return tuple(data_size)


def to_data_size(size, r=all):
    """Converts full size to actual size given range r

    Arguments:
        size (tuple): data size
        r (tuple or all): range specification, ``all`` is full range

    Returns:
        int: data size

    See Also:
        :func:`toDataRange`, :func:`dataSizeFromDataRange`
    """
    dr = to_data_range(size, r=r)
    return int(dr[1] - dr[0])


def to_data_range(size, r=all):
    """Converts range r to numeric range (min,max) given the full array size

    Arguments:
        size (tuple): source data size
        r (tuple or all): range specification, ``all`` is full range

    Returns:
        tuple: absolute range as pair of integers

    See Also:
        :func:`to_data_size`, :func:`dataSizeFromDataRange`
    """

    if r is all:
        return (0, size)

    if isinstance(r, int) or isinstance(r, float):
        r = (r, r + 1)

    if r[0] is all:
        r = (0, r[1])
    if r[0] < 0:
        if -r[0] > size:
            r = (0, r[1])
        else:
            r = (size + r[0], r[1])
    if r[0] > size:
        r = (size, r[1])

    if r[1] is all:
        r = (r[0], size)
    if r[1] < 0:
        if -r[1] > size:
            r = (r[0], 0)
        else:
            r = (r[0], size + r[1])
    if r[1] > size:
        r = (r[0], size)

    if r[0] > r[1]:
        r = (r[0], r[0])

    return r


def convert_h5_to_npy(source, sink):
    with h5py.File(source, "r") as f:
        print(f'converting {source} -> {sink}')
        data = f["Data"]
        np.save(sink, data)


def convert(source, sink, processes=None, verbose=False, **kwargs):
    """Transforms a source into another format.

    Arguments
    ---------
    source : source specification
      The source or list of sources.
    sink : source specification
      The sink or list of sinks.

    Returns
    -------
    sink : sink specification
      The sink or list of sinks.
    """
    source = as_source(source)
    if verbose:
        print(f'converting {source} -> {sink}')
    mod = source_to_module(source)
    if hasattr(mod, 'convert'):
        return mod.convert(source, sink, processes=processes, verbose=verbose, **kwargs)
    else:
        return write(sink, source)


def write(sink, data, *args, **kwargs):
    """Write data to a data source.

    Arguments
    ---------
    sink : str, array, Source class
      The source to write data to.
    data : array
      The data to write to the sink.
    slicing : slice specification or None
      Optional subslice to write data to.

    Returns
    -------
    sink : str, array or Source class
      The sink to which the data was written.
    """
    mod = source_to_module(sink)
    return mod.write(sink, as_source(data), *args, **kwargs)


def convert_stitched_files(raw_directory, **kwargs):
    print("")
    sample_names = ut.get_sample_names(raw_directory, **kwargs)
    for sample_name in sample_names:
        sample_path = os.path.join(raw_directory, sample_name)
        for channel in kwargs["study_params"]["channels_to_stitch"]:
            if kwargs["study_params"]["scanning_system"] == "bruker":
                _, merged_directory = get_bruker_directories(sample_path)
                stitched_file = os.path.join(merged_directory, f"uni_tp-0_ch-{channel}_st-1-x00-y00-1-x00-y01_"
                                                               f"obj-bottom-bottom_cam-bottom_etc.lux.h5")
                stitched_npy = os.path.join(merged_directory, f"stitched_{channel}.npy")
                if not os.path.exists(stitched_npy):
                    ut.print_c(
                        f"[INFO {sample_name}] Converting stitched image (channel {channel}) to numpy format!")
                    convert_h5_to_npy(stitched_file, stitched_npy)
                else:
                    ut.print_c(
                        f"[WARNING {sample_name}] Skipping stitched conversion to npy for channel {channel}: "
                        f"stitched_{channel}.npy file already exists!")
            else:
                stitched_folder = os.path.join(sample_path, f"stitched_{channel}")
                stitched_files = os.path.join(stitched_folder, 'Z<Z,4>.tif')
                stitched_npy = os.path.join(sample_path, f"stitched_{channel}.npy")
                if not os.path.exists(stitched_npy):
                    ut.print_c(f"[INFO {sample_name}] Converting stitched image (channel {channel}) to numpy format!")
                    convert(stitched_files, stitched_npy, processes=None, verbose=False)
                else:
                    ut.print_c(f"[WARNING {sample_name}] Skipping stitched conversion to npy for channel {channel}: "
                               f"stitched_{channel}.npy file already exists!")


def get_bruker_directories(sample_path):
    resolution_directories = [os.path.join(sample_path, i) for i in os.listdir(sample_path)
                              if i.startswith("xy")]
    if len(resolution_directories) == 1:
        resolution_directory = resolution_directories[0]
        merged_directories = [os.path.join(resolution_directory, i) for i in os.listdir(resolution_directory)
                              if i.endswith("merged")]
        if len(merged_directories) == 1:
            merged_directory = merged_directories[0]
            return resolution_directory, merged_directory
        else:
            raise ut.CmliteError(f"More than one merged folder found in {resolution_directory}!")
    else:
        raise ut.CmliteError(f"More than one resolution folder found in {sample_path}!")


def get_sample_directory(raw_directory, sample_name, **kwargs):
    sample_directory = os.path.join(raw_directory, sample_name)
    if kwargs["study_params"]["scanning_system"] == "bruker":
        _, sample_directory = get_bruker_directories(sample_directory)
    return sample_directory
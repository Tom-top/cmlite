import numpy as np

import IO.source as src
import IO.slice as slc
import IO.NPY as npy
import IO.file_utils as fu


###############################################################################
### Source class
###############################################################################

class Source(npy.Source):
    """Memory mapped array source."""

    def __init__(self, location=None, shape=None, dtype=None, order=None, array=None, mode=None, name=None):
        """Memory map source construtor.

        Arguments
        ---------
        array : array
          The underlying data array of this source.
        """
        memmap = _memmap(location=location, shape=shape, dtype=dtype, order=order, mode=mode, array=array)
        super(Source, self).__init__(array=memmap, name=name)

    @property
    def name(self):
        return "Memmap-Source"

    @property
    def array(self):
        """The underlying data array.

        Returns
        -------
        array : array
          The underlying data array of this source.
        """
        return self._array

    @array.setter
    def array(self, value):
        if not isinstance(value, np.memmap):
            array = np.asarray(value)
            value = _memmap(location=self.location, array=array)
        self._array = value

    @property
    def dtype(self):
        """The data type of the source.

        Returns
        -------
        dtype : dtype
          The data type of the source.
        """
        return self._array.dtype

    @dtype.setter
    def dtype(self, value):
        if np.dtype(value) != self.dtype:
            self.array = np.asarray(self.array, dtype=value)

    @property
    def order(self):
        """The order of how the data is stored in the source.

        Returns
        -------
        order : str
          Returns 'C' for C contigous and 'F' for fortran contigous, None otherwise.
        """
        return npy.order(self.array)

    @order.setter
    def order(self, value):
        if value != self.order:
            self.array = np.asarray(self.array, order=value)

    @property
    def location(self):
        """The location where the data of the source is stored.

        Returns
        -------
        location : str or None
          Returns the location of the data source or None if this source lives in memory only.
        """
        return self._array.filename

    @location.setter
    def location(self, value):
        if value != self.location:
            memmap = _memmap(location=value, shape=self.shape, dtype=self.dtype, order=self.order)
            self.array = memmap

    @property
    def offset(self):
        """The offset of the memory map in the file.

        Returns
        -------
        offset : int
          Offset of the memeory map in the file.
        """
        return self._array.offset

    def as_virtual(self):
        return VirtualSource(source=self)

    def as_buffer(self):
        return self._array


class VirtualSource(src.VirtualSource):
    """Virtual memory map source."""

    def __init__(self, source=None, shape=None, dtype=None, order=None, name=None):
        super(VirtualSource, self).__init__(source=source, shape=shape, dtype=dtype, order=order, name=name)

    @property
    def name(self):
        return 'Virtual-Memmap-Source'

    def as_virtual(self):
        return self

    def as_real(self):
        return Source(location=self.location, shape=self.shape, dtype=self.dtype, order=self.order, name=self.name)

    def as_buffer(self):
        return self.as_real().as_buffer()

    @property
    def array(self):
        return self.as_real().array


###############################################################################
### IO Interface
###############################################################################

def is_memmap(source):
    if isinstance(source, (np.memmap, Source)):
        return True
    elif isinstance(source, str):
        if fu.is_file(source):
            try:
                memmap = np.memmap(source)  # analysis:ignore
            except:
                return False
        return True
    else:
        return False


def read(source, slicing=None, mode=None, **kwargs):
    """Write data to a memory map.

    Arguments
    ---------
    sink : str, memmap, or Source
      The sink to write the data to.
    slicing : slice specification
      Optional slice specification of memmap to read from.
    mode : str
      Optional mode spcification of how to open the memmap.

    Returns
    -------
    source : Source
      The read memmap source.
    """

    if isinstance(source, Source):
        if slicing is None:
            return source
        else:
            return source.__getitem__(slicing)

    elif isinstance(source, np.memmap):
        if slicing is None:
            memmap = source
        else:
            memmap = source.__getitem__(slicing)
        return Source(array=memmap)

    elif isinstance(source, str):
        try:
            memmap = _memmap(location=source, mode=mode)
        except:
            raise ValueError('Cannot read memmap from location %r!' % source)

        if slicing is not None:
            memmap = memmap.__getitem__(slicing)

        return Source(array=memmap)

    else:
        raise ValueError('Cannot read memmap from source %r!' % source)


def write(sink, data, slicing=None, **kwargs):
    """Write data to a memory map.

    Arguments
    ---------
    sink : str, memmap, or Source
      The sink to write the data to.
    data : array
      The data to write int the sink.
    slicing : slice specification or None
      Optional slice specification of an existing memmap to write to.

    Returns
    -------
    sink : str, memmap, or Source
      The sink.
    """
    if slc.is_trivial(slicing):
        slicing = (slice(None),)

    if isinstance(sink, (Source, np.memmap)):
        sink.__setitem__(slicing, data.array)

    elif isinstance(sink, str):
        if slicing == (slice(None),):
            memmap = _memmap(location=sink, array=data.array)
        else:
            try:
                memmap = _memmap(location=sink, mode='r+')
            except:
                raise ValueError('Cannot write slice into non-existent memmap at location %r!' % sink)
            memmap.__setitem__(slicing, data.array)

    else:
        raise ValueError('Cannot write memmap to sink %r!' % sink)

    return sink


def create(location=None, shape=None, dtype=None, order=None, mode=None, array=None, as_source=True, **kwargs):
    """Create a memory map.

    Arguments
    ---------
    location : str
      The filename of the memory mapped array.
    shape : tuple or None
      The shape of the memory map to create.
    dtype : dtype
      The data type of the memory map.
    order : 'C', 'F', or None
      The contiguous order of the memmap.
    mode : 'r', 'w', 'w+', None
      The mode to open the memory map.
    array : array, Source or None
      Optional source with data to fill the memory map with.
    as_source : bool
      If True, return as Source class.

    Returns
    -------
    memmap : np.memmap
      The memory map.

    Note
    ----
    By default memmaps are initialized as fortran contiguous if order is None.
    """
    mode = 'w+' if mode is None else mode
    memmap = _memmap(location=location, shape=shape, dtype=dtype, order=order, mode=mode, array=array)
    if as_source:
        return Source(memmap)
    else:
        return memmap


###############################################################################
### Helpers
###############################################################################

def _memmap(location=None, shape=None, dtype=None, order=None, mode=None, array=None):
    """Create a memory map.

    Arguments
    ---------
    location : str
      The filename of the memory mapped array.
    shape : tuple or None
      The shape of the memory map to create.
    dtype : dtype
      The data type of the memory map.
    order : 'C', 'F', or None
      The contiguous order of the memmap.
    mode : 'r', 'w', 'w+', None
      The mode to open the memory map.
    array : array, Source or None
      Optional source with data to fill the memory map with.

    Returns
    -------
    memmap : np.memmap
      The memory map.

    Note
    ----
    By default memmaps are initialized as fortran contiguous if order is None.
    """
    # print location, shape, dtype, order, mode, array
    if isinstance(location, np.memmap):
        array = location
        location = None

    if array is None:
        if not isinstance(location, str):
            raise ValueError('Cannot create memmap without a location!')

        if mode != 'w+' and fu.is_file(location):
            array = np.lib.format.open_memmap(location)

    if array is None:
        if shape is None:
            raise ValueError('Cannot create memmap without shape at location %r!' % location)

        mode = 'w+' if mode is None else mode
        fortran = order in ['F', None]  # default is 'F' for memmaps

        memmap = np.lib.format.open_memmap(location, mode=mode, shape=shape, dtype=dtype, fortran_order=fortran)

    elif isinstance(array, np.memmap):
        location = location if location is not None else array.filename
        location = fu.abspath(location)

        shape = shape if shape is not None else array.shape
        dtype = dtype if dtype is not None else array.dtype
        order = order if order is not None else npy.order(array)

        # if shape != array.shape:
        #  raise ValueError('Shape %r and array shape %r mismatch!' % (shape, array.shape))

        if shape != array.shape or dtype != array.dtype or order != npy.order(array) or location != fu.abspath(
                array.filename):
            fortran = order in ['F', None]  # default is 'F' for memmaps
            memmap = np.lib.format.open_memmap(location, mode='w+', shape=shape, dtype=dtype, fortran_order=fortran)
            if shape == array.shape:
                memmap[:] = array
        else:
            memmap = array

        if mode is None:
            mode = 'r+'
        if mode != memmap.mode:
            memmap = np.lib.format.open_memmap(location, mode=mode)

    elif isinstance(array, np.ndarray):
        if not isinstance(location, str):
            raise ValueError('Cannot create memmap without a location!')

        shape = shape if shape is not None else array.shape
        dtype = dtype if dtype is not None else array.dtype
        order = order if order is not None else npy.order(array)

        if shape != array.shape:
            raise ValueError('Shape %r and array shape %r mismatch!' % (shape, array.shape))

        fortran = order in ['F', None]  # default is 'F' for memmaps
        memmap = np.lib.format.open_memmap(location, mode='w+', shape=shape, dtype=dtype, fortran_order=fortran)
        memmap[:] = array

        if mode is None:
            mode = 'r+'
        if mode != memmap.mode:
            memmap = np.lib.format.open_memmap(location, mode=mode)

    else:
        raise ValueError('Array is not a valid!')

    return memmap


def header_size(filename):
    """Return the offset of a header in a memmaped file.

    Arguments
    ---------
    filename : str
      Filename of the npy fie.

    Returns
    -------
    offset : int
      The offest due to the header.
    """
    with open(filename, 'rb') as f:
        major, minor = np.lib.format.read_magic(f)
        shape, fortran, dtype = np.lib.format.read_array_header_1_0(f)
        offset = f.tell()

    return offset
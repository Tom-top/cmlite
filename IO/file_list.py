import os
import re
import glob
import natsort
import warnings
import itertools
import numbers

import numpy as np
import multiprocessing as mp
import concurrent.futures

import IO.IO as io
import IO.file_utils as fu
import IO.source as src
import IO.slice as slc

import utils.utils as ut
import utils.tag_expression as te

import parallel_processing.parallel_traceback as ptb


class Source(src.VirtualSource):
    """File list source.

    Note
    ----
    The full shape of the file list source is the shape of the expression and
    the shape of the data in each file, i.e. shape = file_list_shape + array_shape.
    """

    def __init__(self, expression=None, file_list=None, axes_order=None, shape=None, dtype=None, order=None, name=None):
        """File list source class construtor.

        Arguments
        ---------
        expression : str or Expression
          The expression specifying a file list.
        file_list : list of strs
          List of filenames.
        axes_order : list of str
          List of names indicating the ordering of the tags along the axes.
        name : str or None
         Optional name of the source.

        Note
        ----
        Either expression or file_list need to be specified.
        """
        super(Source, self).__init__(name=name)
        expression, file_list = _expression_or_file_list(expression=expression, file_list=file_list)

        # properties
        self._expression = expression
        self._file_list = file_list
        self._axes_order = axes_order
        self._shape = shape
        self._dtype = dtype
        self._order = order

    @property
    def name(self):
        return "FileList-Source"

    @property
    def file_list(self, sort=True):
        """The underlying file list.

        Returns
        -------
        filelist : list
          The underlying sources of this source.
        """
        if self._file_list is None:
            self._file_list = _file_list(expression=self.expression, sort=sort)
        return self._file_list

    @file_list.setter
    def file_list(self, value):
        raise ValueError('Cannot set file_list for this source!')

    @property
    def expression(self):
        """The underlying expression of this file list.

        Returns
        -------
        expression : str
          The underlying expression of this source.
        """
        if self._expression is None:
            self._expression = te.detect(self.file_list)
        return self._expression

    @expression.setter
    def expression(self, value):
        raise ValueError('Cannot set expression for this source!')

    @property
    def axes_order(self):
        """Optional ordering of the tag names.

        Returns
        -------
        axes_order : list of str
          The ordered axis names.
        """
        tag_names = self.expression.tag_names()
        axes_order = self._axes_order
        if axes_order is None:
            axes_order = []
        for a in axes_order:
            if a not in tag_names:
                raise ValueError('Axes name %s is not in the tags %r!' % (a, tag_names))
        axes_order = axes_order + [n for n in tag_names if n not in axes_order]
        return axes_order

    @axes_order.setter
    def axes_order(self, value):
        self._axes_order = value

    @property
    def shape(self, axis=None):
        """The shape of the source.

        Returns
        -------
        shape : tuple
          The shape of the source.
        """
        if self._shape is None:  # cache the result
            self._shape = shape(expression=self.expression, file_list=self.file_list, axes_order=self.axes_order,
                                axis=axis)
        return self._shape

    @shape.setter
    def shape(self, value):
        self._shape = value

    @property
    def dtype(self):
        """The data type of the source.

        Returns
        -------
        dtype : dtype
          The data type of the source.
        """
        if self._dtype is None:
            self._dtype = dtype(expression=self._expression, file_list=self.file_list)
        return self._dtype

    @property
    def element_strides(self):
        # TODO:
        raise NotImplementedError()

    @property
    def ndim_file(self):
        """Source dimension of the individual files."""
        return len(self.shape_file)

    @property
    def ndim_list(self):
        """Source dimensions of the file list."""
        return len(self.shape_list)

    @property
    def shape_file(self):
        """Source shape of the individual files."""
        return shape_file(expression=self._expression, file_list=self._file_list)

    @property
    def shape_list(self):
        """Source shape of the file list."""
        return shape_list(expression=self._expression, file_list=self._file_list, axes_order=self.axes_order)

    @property
    def axes_file(self):
        """Source axes that constitute individual file dimensions in the full array."""
        return tuple(i for i in range(self.ndim_file))

    @property
    def axes_list(self):
        """Source axes that constitute the dimensions of the file list in the full array."""
        ndim_file = self.ndim_file
        return tuple(ndim_file + i for i in range(self.ndim_list))

    def tag_to_axes_order(self):
        """Map from the tag list from the file expression and the axes of this source."""
        tag_names = self.expression.tag_names()
        axes_order = self.axes_order
        tag_to_axes = []
        for n in axes_order:
            for i, m in enumerate(tag_names):
                if n == m:
                    tag_to_axes.append(i)
                    break
        return tag_to_axes

    def axes_to_tag_order(self):
        """Map from axes of this source to the tag list from the file expression."""
        tag_names = self.expression.tag_names()
        axes_order = self.axes_order
        axes_to_tags = []
        for n in tag_names:
            for i, m in enumerate(axes_order):
                if n == m:
                    axes_to_tags.append(i)
                    break
        return axes_to_tags

    def __getitem__(self, slicing, processes=None, order=None):
        e = self.expression

        shape = self.shape
        ndim = self.ndim
        ndim_list = e.ntags()

        slicing = slc.unpack_slicing(slicing, ndim)

        slicing_file = slicing[:-ndim_list]
        slicing_list = slicing[-ndim_list:]

        shape_file = shape[:-ndim_list]
        shape_list = shape[-ndim_list:]

        sliced_shape_file = slc.sliced_shape(slicing=slicing_file, shape=shape_file)
        # sliced_shape_list = slc.sliced_shape(slicing=slicing_list, shape=shape_list)

        # start indices
        indices_start = self.expression.indices(self.file_list[0])
        # print(indices_start)
        # TODO: steps in file list

        # genereate file list to read
        # Note: indices increase according to the axes order but their own order is in tag order
        indices = []
        slicing_list_indices = []
        shape_list_keep_dims = ()
        slicing_keep_dims_to_final = (Ellipsis,)
        for sl, s, i in zip(slicing_list, shape_list, indices_start):
            if isinstance(sl, slice):
                slice_indices = sl.indices(s)
                slice_indices = (slice_indices[0] + i, slice_indices[1] + i, slice_indices[2])
                indices.append(range(*slice_indices))
                n = len(indices[-1])
                slicing_list_indices.append(range(n))
                shape_list_keep_dims += (n,)
                slicing_keep_dims_to_final += (slice(None),)
            elif isinstance(sl, (list, np.ndarray)):
                indices.append(np.array(sl) + i)
                n = len(indices[-1])
                slicing_list_indices.append(sl)
                shape_list_keep_dims += (n,)
                slicing_keep_dims_to_final += (slice(None),)
            elif isinstance(sl, numbers.Integral):
                indices.append([sl + i])
                slicing_list_indices.append([0])
                shape_list_keep_dims += (1,)
                slicing_keep_dims_to_final += (0,)
            else:
                raise IndexError('Invalid slice specification %r!' % sl)
        indices.reverse()
        indices = itertools.product(*indices)
        indices = [i[::-1] for i in indices]
        slicing_list_indices.reverse()
        slicing_list_indices = itertools.product(*slicing_list_indices)
        slicing_list_indices = [i[::-1] for i in slicing_list_indices]
        # print(indices, slicing_list_indices, slicing_keep_dims_to_final)

        axes_to_tags = self.axes_to_tag_order()
        if len(axes_to_tags) > 1 and axes_to_tags != list(range(len(axes_to_tags))):
            indices = [tuple(i[j] for j in axes_to_tags) for i in indices]

        fl = [e.string_from_index(i) for i in indices]
        # print(fl)

        dtype = self.dtype

        data = np.zeros(sliced_shape_file + shape_list_keep_dims, dtype=dtype, order=order)

        # @ptb.parallel_traceback
        def func(filename, index, data=data, slicing=slicing_file):
            index = (Ellipsis,) + index
            data[index] = io.read(filename, slicing=slicing, processes='serial')

        if processes is None:
            processes = mp.cpu_count()

        if processes == 'serial':
            for f, i in zip(fl, slicing_list_indices):
                func(f, i)
        else:
            with concurrent.futures.ThreadPoolExecutor(processes) as executor:
                executor.map(func, fl, slicing_list_indices)

        data = data[slicing_keep_dims_to_final]

        return data

    def __setitem__(self, slicing, data, processes=None):
        e = self.expression

        shape = self.shape
        ndim = self.ndim
        ndim_list = e.ntags()

        slicing = slc.unpack_slicing(slicing, ndim)

        slicing_file = slicing[:-ndim_list]
        slicing_list = slicing[-ndim_list:]

        shape_list = shape[-ndim_list:]

        # start indices
        indices_start = self.expression.indices(self.file_list[0])
        # TODO: steps in file list

        # genereate file list to read
        # Note: indices increase according to the axes order but thier own order is in tag order
        indices = []
        for sl, s, i in zip(slicing_list, shape_list, indices_start):
            if isinstance(sl, slice):
                slice_indices = sl.indices(s)
                slice_indices = (slice_indices[0] + i, slice_indices[1] + i, slice_indices[2])
                indices.append(range(*slice_indices))
            elif isinstance(sl, (list, np.ndarray)):
                indices.append(np.array(sl) + i)
            elif isinstance(sl, numbers.Integral):
                indices.append([sl + i])
            else:
                raise IndexError('Invalid slice specification %r!' % sl)
        indices.reverse()
        indices = itertools.product(*indices)
        indices = [i[::-1] for i in indices]

        axes_to_tags = self.axes_to_tag_order()
        if len(axes_to_tags) > 1 and axes_to_tags != list(range(len(axes_to_tags))):
            indices = [tuple(i[j] for j in axes_to_tags) for i in indices]

        fl = [e.string_from_index(i) for i in indices]
        # print indices, fl

        # create directory if it does not exists
        # Note: move this to func if files need to be distributed accross several directories
        fu.create_directory(fl[0], split=True)

        if processes is None:
            processes = mp.cpu_count()

        @ptb.parallel_traceback
        def func(filename, index, data=data, slicing=slicing_file):
            index = (Ellipsis,) + index
            io.write(sink=filename, data=data[index], slicing=slicing, processes='serial')

        if processes == 'serial':
            for f, i in zip(fl, indices):
                func(f, i)
        else:
            with concurrent.futures.ThreadPoolExecutor(processes) as executor:
                executor.map(func, fl, indices)

    @property
    def array(self):
        return self._getitem__(slice(None))

    def __str__(self):
        try:
            name = self.name
            name = '%s' % name if name is not None else ''
        except:
            name = ''

        try:
            shape = self.shape  # _shape
            shape = '%r' % ((shape,)) if shape is not None else ''
        except:
            shape = ''

        try:
            dtype = self.dtype  # _dtype
            dtype = '[%s]' % dtype if dtype is not None else ''
        except:
            dtype = ''

        try:
            order = self.order  # _order
            order = '|%s|' % order if order is not None else ''
        except:
            order = ''

        try:
            file_list = '<%d>' % len(self._file_list)
        except:
            file_list = ''

        try:
            expression = self.expression.tag()
            if len(expression) > 100:
                expression = expression[:50] + '...' + expression[-50:]
            expression = '{%s}' % expression
        except:
            expression = ''

        return name + shape + dtype + file_list + expression

    def as_real(self):
        return self

    def as_virtual(self):
        return VirtualSource(expression=self.expression, file_list=None,
                             shape=self._shape, dtype=self._dtype, order=self._order,
                             axes_order=self._axes_order)

    def as_buffer(self):
        return self.array


class VirtualSource(src.VirtualSource):
    """Virtual file list source."""

    def __init__(self, expression=None, file_list=None, shape=None, dtype=None, order=None, axes_order=None,
                 source=None, name=None):
        super(VirtualSource, self).__init__(source=source, shape=shape, dtype=dtype, order=order, name=name)
        self._expression = expression
        self._file_list = file_list
        self._axes_order = axes_order

    @property
    def name(self):
        return 'Virtual-FileList-Source'

    @property
    def expression(self):
        """The underlying expression of this file list.

        Returns
        -------
        expression : str
          The underlying expression of this source.
        """
        if self._expression is None:
            self._expression = te.detect(self.file_list)
        return self._expression

    @expression.setter
    def expression(self, value):
        raise ValueError('Cannot set expression for this source!')

    @property
    def axes_order(self):
        return self._axes_order

    @axes_order.setter
    def axes_order(self, value):
        raise ValueError('Cannot set axes_order for this source!')

    @property
    def file_list(self):
        return self._file_list

    @file_list.setter
    def file_list(self, value):
        raise ValueError('Cannot set file_list for this source!')

    @property
    def shape(self, axis=None):
        """The shape of the source.

        Returns
        -------
        shape : tuple
          The shape of the source.
        """
        if self._shape is None:  # cache the result
            self._shape = shape(expression=self.expression, file_list=self.file_list, axes_order=self.axes_order,
                                axis=axis)
        return self._shape

    @shape.setter
    def shape(self, value):
        self._shape = value

    @property
    def dtype(self):
        """The data type of the source.

        Returns
        -------
        dtype : dtype
          The data type of the source.
        """
        if self._dtype is None:
            self._dtype = dtype(expression=self._expression, file_list=self.file_list)
        return self._dtype

    @property
    def element_strides(self):
        # TODO:
        raise NotImplementedError()

    def as_virtual(self):
        return self

    def as_real(self):
        return Source(expression=self.expression, file_list=self.file_list, axes_order=self.axes_order,
                      shape=self.shape, dtype=self.dtype, order=self.order, name=self.name)

    def as_buffer(self):
        return self.as_real().as_buffer()

    @property
    def array(self):
        return self.as_real().array

    def __str__(self):
        try:
            name = self.name
            name = '%s' % name if name is not None else ''
        except:
            name = ''

        try:
            shape = self.shape  # _shape
            shape = '%r' % ((shape,)) if shape is not None else ''
        except:
            shape = ''

        try:
            dtype = self.dtype  # _dtype
            dtype = '[%s]' % dtype if dtype is not None else ''
        except:
            dtype = ''

        try:
            order = self.order  # _order
            order = '|%s|' % order if order is not None else ''
        except:
            order = ''

        try:
            file_list = '<%d>' % len(self._file_list)
        except:
            file_list = ''

        try:
            expression = self.expression.tag()
            if len(expression) > 100:
                expression = expression[:50] + '...' + expression[-50:]
            expression = '{%s}' % expression
        except:
            expression = ''

        return name + shape + dtype + file_list + expression


def is_file_list(expression, exists=False, tag_names=None, n_tags=-1, verbose=False):
    """Checks if the expression is a valid file list.

    Arguments
    ---------
    expression : str
      The regular expression to check.
    exists : bool
      If True, check if at least one file exists.
    tag_names :  list of str or None
      List of tag names expected to be present in the expression.
    n_Tags : int or None
      Number of tags to expect.
    verbose : bool
      If True, print reason why the epxression does not represent the desired file list.

    Returns
    -------
    is_expression : bool
      Returns True if the expression fullfills the desired criteria and at least one file matching the expression exists.
    """
    if isinstance(expression, Source):
        return True

    if not isinstance(expression, (str, te.Expression)):
        if verbose:
            warnings.warn('The expression %r is not a string or valid Source!' % expression)
        return False

    if fu.is_directory(expression):
        if exists:
            if len(os.listdir(expression)) == 0:
                if verbose:
                    warnings.warn('No files exists in the directory %s!' % expression)
                return False
            else:
                return True
        else:
            return True

    if tag_names is not None or n_tags is not None:
        t = te.Expression(expression) if not isinstance(expression, te.Expression) else expression
        if n_tags is not None:
            if n_tags < 0 and -n_tags > t.ntags():
                if verbose:
                    warnings.warn('Expression has not required number %d of tags, but %d!' % (n_tags, t.ntags()))
                return False
            elif n_tags >= 0 and n_tags != t.ntags():
                if verbose:
                    warnings.warn('Expression has not required number %d of tags, but %d!' % (n_tags, t.ntags()))
                return False

        if tag_names is not None:
            if tag_names != t.tag_names():
                if verbose:
                    warnings.warn('Expression has not required tags %r, but %r!' % (tag_names, t.tag_names()))
                return False

    if exists:
        f = _first_file(expression)
        if f is None:
            if verbose:
                warnings.warn('Expression does not point to any files!')
            return False

    return True


def ndim(expression=None, file_list=None):
    """Calculates the dimension of the file list given by an expression.

    Arguments
    ---------
    expression : str
      The expression for the file_list.
    file_list : list
      Optional file_list to speed up calculation.

    Returns
    -------
    ndim : int
      The dimension of the file list given by the expression.
    """
    expression, file_list = _expression_and_file_list(expression=expression, file_list=file_list)

    if len(file_list) == 0:
        raise ValueError('Cannot determine dimension of the file list %r without files.!' % expression)

    return io.ndim(file_list[0]) + expression.ntags()


# TODO: arbitrary axes mixing file and list dimensions
def shape(expression=None, file_list=None, axes_order=None, axis=None):
    """Calculates the shape of the data in a file list.

    Arguments
    ---------
    expression : str or None
      The regular epression for the file list.
    file_list : list or None
      List of files.
    axis : int or None
      The shape along a specific axis. Can speed up the shape calculation.
    axes_order : list or None
      The names of how to order the different tag names in the expression.
      If None, use ordering of the tags in the expression.

    Returns
    -------
    shape : int or tuple of ints
      The shape of the array st  ored in a file list.
    """
    expression, file_list = _expression_and_file_list(expression=expression, file_list=file_list)

    shapelist = shape_list(expression=expression, file_list=file_list, axes_order=axes_order)

    if axis is not None:
        if axis < 0 and -axis < len(shapelist):
            return shapelist[axis]

    # determine dimensions in each file
    shapefile = shape_file(expression=expression, file_list=file_list)

    # full shape
    shape_full = shapefile + shapelist

    if axis is not None:
        return shape_full[axis]
    else:
        return shape_full


def shape_file(expression=None, file_list=None):
    """Calculates the shape of the data in a file list.

    Arguments
    ---------
    expression : str or None
      The regular epression for the file list.
    file_list : list or None
      List of files.

    Returns
    -------
    shape : int or tuple of ints
      The shape of the array st  ored in a file list.
    """
    expression, file_list = _expression_and_file_list(expression=expression, file_list=file_list)

    if len(file_list) == 0:
        raise ValueError('Cannot determine dimension of the file list %r without files.!' % expression)

    # determine dimensions in each file
    shape_file = io.shape(file_list[-1])  # Take the last because the first has Z

    return shape_file


def shape_list(expression=None, file_list=None, axes_order=None):
    """Calculates the shape of the data in a file list.

    Arguments
    ---------
    expression : str or None
      The regular epression for the file list.
    file_list : list or None
      List of files.

    axes_order : list or None
      The names of how to order the different tag names in the expression.
      If None, use ordering of the tags in the expression.

    Returns
    -------
    shape : int or tuple of ints
      The shape of the array along the dimensions created by the file list.
    """
    expression, file_list = _expression_and_file_list(expression=expression, file_list=file_list)

    if len(file_list) == 0:
        raise ValueError('Cannot determine dimension of the file list %r without files.!' % expression)

    # ordering of the axes
    if axes_order is None:
        axes_order = expression.tag_names()

    # determine dimensions along the file list
    values0 = expression.values(file_list[0])
    if len(axes_order) == 1:
        shape_list = (len(file_list),)
    else:
        shape_list = ()
        for a in axes_order:
            values = values0.copy()
            values.__delitem__(a)
            e = te.Expression(expression.string(values=values))
            search = re.compile(e.re()).search
            shape_list += (len([f for f in file_list if search(f)]),)

    return shape_list


def dtype(expression=None, file_list=None):
    """Returns data type of the array stored in a file list.

    Arguments
    ---------
    expression : str
      The regular epression for the file list.

    Returns
    -------
    dtype : dtype
      The data type of the file list.
    """
    file_list = _file_list(expression=expression, file_list=file_list)

    if len(file_list) == 0:
        raise ValueError('Cannot determine dtype from file list %r without files!' % expression)

    return io.dtype(file_list[0])


def order(expression=None, file_list=None):
    """Returns order of the array stored in a file list.

    Arguments
    ---------
    expression : str
      The regular epression for the file list.

    Returns
    -------
    dtype : dtype
      The data type of the file list.
    """
    file_list = _file_list(expression=expression, file_list=file_list)

    if len(file_list) == 0:
        raise ValueError('Cannot determine order from file list %r without files!' % expression)

    order_file = io.order(file_list[0])
    if order_file == 'F':
        return 'F'
    else:
        return None


# TODO:
def read(source, slicing=None, axes_order=None, **kwargs):
    raise NotImplementedError('read for FileList not implemented yet!')


def write(sink, data, slicing=None, axes_order=None, processes=None, **kwargs):
    raise NotImplementedError('write for FileList not implemented yet!')


def create(location=None, shape=None, dtype=None, array=None, as_source=True):
    raise NotImplementedError('create for FileList not implemented yet!')


def _expression_or_file_list(expression=None, file_list=None):
    if isinstance(expression, te.Expression) or expression is None:
        pass
    elif fu.is_directory(expression):
        file_list = _file_list(expression=expression, sort=True)
        expression = None
    elif isinstance(expression, str):
        expression = te.Expression(expression)
    else:
        raise ValueError('The expression %r is not valid!' % expression)

    if file_list is not None and not isinstance(file_list, list):
        raise ValueError('The file_list %r is not a list or None!' % file_list)

    if expression is None and file_list is None:
        raise ValueError('Expresson and file_list cannot both be None!')

    return expression, file_list


def _file_list(expression=None, file_list=None, sort=True, verbose=False):
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
    if isinstance(file_list, list):
        return file_list

    if isinstance(expression, te.Expression):
        fl = glob.glob(expression.glob())
    elif fu.is_directory(expression):
        expression = fu.join(expression, '*')
        fl = glob.glob(expression)
    else:
        e = te.Expression(expression)
        fl = glob.glob(e.glob())

    if verbose and len(fl) == 0:
        warnings.warn('No files found matching %s !' % expression)
        return []

    if sort:
        fl = natsort.natsorted(fl)

    return fl


def _first_file(expression):
    fl = _file_list(expression=expression, file_list=None, sort=False, verbose=False)
    if len(fl) > 0:
        return fl[0]
    else:
        return None


def _expression_and_file_list(expression=None, file_list=None):
    if isinstance(expression, te.Expression) or expression is None:
        pass
    elif fu.is_directory(expression):
        if file_list is None:
            file_list = glob.glob(fu.join(expression, '*'))
        expression = te.detect(file_list)
    elif isinstance(expression, (str, list)):
        expression = te.Expression(expression)
    else:
        raise ValueError('Expression %r is not valid!' % expression)

    if file_list is None:
        if expression is None:
            raise ValueError('Either expression or file_list need to be specified!')
        file_list = glob.glob(expression.glob())
    elif isinstance(file_list, list):
        if expression is None:
            expression = te.detect(file_list)
    else:
        raise ValueError('The file_list %r is not value!' % file_list)

    return expression, file_list


def _expression_or_file_list(expression=None, file_list=None):
    if isinstance(expression, te.Expression) or expression is None:
        pass
    elif fu.is_directory(expression):
        file_list = _file_list(expression=expression, sort=True)
        expression = None
    elif isinstance(expression, str):
        expression = te.Expression(expression)
    else:
        raise ValueError('The expression %r is not valid!' % expression)

    if file_list is not None and not isinstance(file_list, list):
        raise ValueError('The file_list %r is not a list or None!' % file_list)

    if expression is None and file_list is None:
        raise ValueError('Expresson and file_list cannot both be None!')

    return expression, file_list


def convert(source, sink, processes=None, verbose=False):
    """Converts list of files to a sink in parallel

    Arguments
    ---------
    source : Source
      File list source.
    sink : Source
      A sink to write the source to.

    Returns
    -------
    sink : Source
      The sink the data was converted to.
    """

    # read files
    if not isinstance(source, Source):
        raise ValueError('Source should be a FileList source, found %r!' % source)

    expression = source.expression
    shape = source.shape
    dtype = source.dtype
    shape_list = source.shape_list
    file_list = source.file_list

    # genereate file lists and slicings
    indices_file_start = expression.indices(file_list[0])
    indices_slice = [np.arange(s) for s in shape_list]
    indices_file = [s + i for i, s in zip(indices_file_start, indices_slice)]

    indices_file.reverse()
    indices_file = itertools.product(*indices_file)
    indices_file = [i[::-1] for i in indices_file]

    indices_slice.reverse()
    indices_slice = itertools.product(*indices_slice)
    indices_slice = [i[::-1] for i in indices_slice]

    axes_to_tags = source.axes_to_tag_order()
    if len(axes_to_tags) > 1 and axes_to_tags != list(range(len(axes_to_tags))):
        indices_file = [tuple(i[j] for j in axes_to_tags) for i in indices_file]
    file_list = [expression.string_from_index(i) for i in indices_file]

    sink = io.create(sink, shape=shape, dtype=dtype)
    sink_virtual = sink.as_virtual()

    if processes is None:
        processes = mp.cpu_count()

    @ptb.parallel_traceback
    def _convert(filename, index_slicing, sink=sink_virtual, verbose=verbose):
        slicing = (Ellipsis,) + index_slicing
        ut.print_c(f'[INFO] Progress: slice {index_slicing[0]}/{shape[-1]}', end="\r")
        sink.as_real()[slicing] = io.read(filename, processes='serial')

    if processes == 'serial':
        for f, i in zip(file_list, indices_slice):
            _convert(f, i)
    else:
        with concurrent.futures.ThreadPoolExecutor(processes) as executor:
            executor.map(_convert, file_list, indices_slice)

    return sink


def split_file_expression(filename):
    """Split the regular expression at the digit place holder

    Arguments:
        filename (str): file name as regular expression

    Returns:
        tuple: file header, file extension, digit format
    """

    search_regex = re.compile('.*\\\\d\{(?P<digit>\d)\}.*').search
    m = search_regex(filename)

    if not m is None:
        digits = int(m.group('digit'))
        search_regex = re.compile('.*(?P<replace>\\\\d\{\d\}).*').search
        m = search_regex(filename)
        fs = filename.split(m.group('replace'))
        if not len(fs) == 2:
            raise RuntimeError('FileList: more than a single replacement for z indexing!')
        fileheader = fs[0]
        fileext = fs[1]

    else:  # fileheader is given
        digits = 4
        fileheader = filename
        fileext = '.tif'

    digitfrmt = "%." + str(digits) + "d"

    return (fileheader, fileext, digitfrmt)


def file_expression_to_file_name(filename, z):
    """Insert a number into the regular expression

    Arguments:
        filename (str): file name as regular expression
        z (int): z slice index

    Returns:
        str: file name
    """

    (fileheader, fileext, digitfrmt) = split_file_expression(filename)
    return fileheader + (digitfrmt % z) + fileext



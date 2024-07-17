import os
import re
import shutil
import glob
import platform

import json
import numpy as np
from lxml import etree
from natsort import natsorted
from PIL import Image
from PIL.ExifTags import TAGS

import IO.IO as io
import IO.file_list as fl
import IO.file_utils as fu

import utils.utils as ut
import settings

# avoid 'TERM enviroment variable not set' message
os.environ["TERM"] = 'dumb'

terastitcher_binary = None
"""str: the TeraStitcher executable
Notes:
    - setup in :func:`initializeStitcher`
"""

initialized = False
"""bool: True if the TeraStitcher binary is setup
Notes:
    - setup in :func:`initializeStitcher`
"""


def initialize_stitcher(path=None):
    """Initialize settings for the TeraStitcher
    Arguments:
        path (str or None): path to TeraStitcher root directory, if None
        :const:`ClearMap.Settings.TeraStitcherPath` is used.

    See also:
        :const:`TeraStitcherBinary`,
        :const:`Initialized`
    """

    global terastitcher_binary, initialized

    if path is None:
        path = settings.terastitcher_path

    # search for elastix binary
    terasticher_bin_path = os.path.join(path, 'bin/terastitcher')
    if os.path.exists(terasticher_bin_path):
        terastitcher_binary = terasticher_bin_path
    else:
        raise RuntimeError(
            "Cannot find TeraSticher binary %s, set path in Settings.py accordingly!" % terastitcher_binary)

    # add the global optimization script folder
    if not fu.is_file(os.path.join(path, 'LQP_HE.py')):
        print('Warning: the global optimization file %s for TeraStitcher cannot be found in %s' % ('LQP_HE.py', path))
    os.environ["__LQP_PATH__"] = path

    initialized = True

    print("TeraSticher sucessfully initialized from path: %s" % path)

    return path


initialize_stitcher()


def check_sticher_initialized():
    """Checks if TeraSticher is initialized

    Returns:
        bool: True if TeraSticher paths are set.
    """

    global initialized
    if not initialized:
        raise RuntimeError(
            "TeraSticher not initialized: run initializeTeraSticher(path) with proper path to TeraSticher first")
    return True


def find_first_file(filename):
    """Returns the first file that matches the regular expression

    Arguments:
        filename (str): file name as regular expression
  placefile = st.placeTiles(thresfile, xmlResultFile = os.path.join(RawDataDirectory, 'TeraStitcher_place.xml'), algorithm = 'LQP')
  arbirary file that matched first is returned

    Returns:
        string or None: first file name that matches the regular expression or None if no match found

    Note:

      For large file numbers speed can be imporved by setting the option sort = False
    """

    def list_dir(path):
        for f in natsorted(os.listdir(path)):
            if not f.startswith('.'):
                yield f

    # split full paths in case path contains regular expression
    fnsp = filename.split(os.path.sep)
    print(fnsp)

    # handle absolute path
    if fnsp[0] == '':  # aboslute path
        fn = os.path.sep
        fnsp.pop(0)
    else:  # relative path
        # if absolute:
        #  fn = os.path.abspath(os.curdir)
        # else:
        fn = '.'
    # no group info
    for n in fnsp:
        search = re.compile('^' + n + '$').search
        match = None
        for x in list_dir(fn):
            if search(x):
                match = x
                print(match)
                break
        if match is not None:
            if fn.endswith(os.path.sep):
                fn = fn + match
            else:
                fn = fn + os.path.sep + match
        else:
            return None
    return fn


def find_file_info(filename):
    """Tries to infer relevant information from filename for tiling / stitching

    Arguments:
      filename (str): filename to infer information from

    Returns:
      dict: dictionary with relavant information: resolution (microns per pixel), overlap (microns), size (pixel)

    Note:
      resolution is in microns per pixel, overlap is in microns and size is in pixel
    """

    # TODO: move this to the individual file readers

    def ome_info(fn):
        ret = {}
        i = Image.open(fn)
        Dict = {}
        for m, n in zip(i.tag.keys(), i.tag.values()):
            Dict[m] = n
        # info = i.tag.as_dict()
        for tag, value in Dict.items():
            decoded = TAGS.get(tag)
            ret[decoded] = value
        return ret

    imginfo = ome_info(filename)
    keys = imginfo.keys()

    finfo = {'resolution': None, 'overlap': None, 'size': None}

    # get image sizes
    if ('ImageHeight' in keys) and ('ImageWidth' in keys):
        finfo['size'] = (imginfo['ImageWidth'], imginfo['ImageHeight'])
    else:
        finfo['size'] = io.data_size(filename)

    if 'ImageDescription' in keys:
        imgxml = imginfo['ImageDescription']

        if isinstance(imgxml, tuple):
            imgxml = imgxml[0]

        # print(imgxml[0])
        # imgxml.encode('ascii')

        # print(imgxml)

        # imgxml = "'''"+imgxml+"'''"

        #    print(imgxml.encode("utf-8"))

        #    print(imgxml)

        #    imgxml2 = imgxml.encode("utf-8")
        ##    print(imgxml2)
        #    text_file = open(os.path.join("/home/thomas.topilko/Desktop", "Output.txt"), "w")
        #    text_file.write(imgxml2)
        #    text_file.close()

        try:
            imgxml = etree.fromstring(imgxml.encode("utf-8"))  # "ascii"
        except:
            imgxml = _cleanXML(
                imgxml)  # for large images the TileConfiguration entry is huge and can cause a AttValue error in the xml parser
            imgxml = etree.fromstring(imgxml.encode("utf-8"))

        # get resolution
        pix = [x for x in imgxml.iter('{*}Pixels')]
        if len(pix) > 0:
            pix = pix[0].attrib
            keys = pix.keys()
            if 'PhysicalSizeX' in keys and 'PhysicalSizeY' in keys and 'PhysicalSizeZ' in keys:
                finfo['resolution'] = (
                    float(pix['PhysicalSizeX']), float(pix['PhysicalSizeY']), float(pix['PhysicalSizeZ']))

        # overlapX = [x.attrib["Value"] for x in imgxml.iter('{*}prop') if x.attrib["label"]=="xyz-Table X Overlap"]
        overlapX = [x.attrib["Value"] for x in imgxml.iter('{*}xyz-Table_X_Overlap')]
        # overlapY = [x.attrib["Value"] for x in imgxml.iter('{*}prop') if x.attrib["label"]=="xyz-Table Y Overlap"]
        overlapY = [x.attrib["Value"] for x in imgxml.iter('{*}xyz-Table_Y_Overlap')]

        finfo['overlap'] = (float(overlapX[0]), float(overlapY[0]))

    return finfo


def _cleanXML(xmlstring):
    """Removes the TileConfiguration entry in the xml string that gets too large for too many tiles"""

    start_str = "<TileConfiguration"
    end_str = "/>"

    start = xmlstring.find(start_str)
    end = xmlstring.find(end_str, start)

    return xmlstring[:start] + xmlstring[(end + len(end_str)):]


def find_file_list(filename, sort=True, groups=None, absolute=True):
    """Returns the list of files that match the regular expression (including variable paths)

    Arguments:
        filename (str): file name as regular expression
        sort (bool): naturally sort the list
        groups (list or None): if list also return the values found for the indicated group identifier

    Returns:
        list: file names
        list: if groups is not None, a list of lists of group identifier
    """

    if sort:
        def list_dir(path):
            for f in natsorted(os.listdir(path)):
                if not f.startswith('.'):
                    yield f
    else:
        def list_dir(path):
            for f in os.listdir(path):
                if not f.startswith('.'):
                    yield f

    # split full paths in case path contains regular expression
    if platform.system().lower() == "windows":
        pattern = re.compile(r'stack_\\\[.*\\\]_3\.tif')
        match = pattern.search(filename)
        if match:
            start, end = match.span()
            # directory_part = filename[:start]
            fnsp = ["", filename]  # Fixme: THIS IS BROKEN
        else:
            print("No match found for the regular expression in the path.")
    else:
        fnsp = filename.split(os.path.sep)

    # handle absolute path
    if fnsp[0] == '':  # absolute path
        files = [os.path.sep]
        fnsp.pop(0)
    else:  # relative path
        if absolute:
            files = [os.path.abspath(os.curdir)]
        else:
            files = ['.']

    # handle pure directory expression -> all files
    if len(fnsp) > 0 and fnsp[-1] == '':
        fnsp[-1] = '.*'

    if groups is None:
        # no group info
        for n in fnsp:
            search = re.compile(n).search
            newfiles = []
            for f in files:
                matches = map(search, list_dir(f))
                matches = [x for x in matches if x]
                for m in matches:
                    if f.endswith(os.path.sep):
                        newfiles.append(f + m.string)
                    else:
                        newfiles.append(f + os.path.sep + m.string)
            files = newfiles

        return files

    else:  # with group info
        infos = [[None for x in groups]]
        for n in fnsp:
            search = re.compile(n).search
            newfiles = []
            newinfos = []

            for f, i in zip(files, infos):
                matches = map(search, list_dir(f))
                matches = [x for x in matches if x]
                for m in matches:
                    if f.endswith(os.path.sep):
                        newfiles.append(f + m.string)
                    else:
                        newfiles.append(f + os.path.sep + m.string)

                    d = m.groupdict()
                    ii = list(i)
                    for k, v in d.items():
                        try:
                            j = groups.index(k)
                            ii[j] = v
                        except:
                            pass
                    newinfos.append(ii)

            files = newfiles
            infos = newinfos

        return np.array(files), np.array(infos)


def displacements(size, resolution=(1.0, 1.0), overlap=(0.0, 0.0), addOverlap=0, units='Microns'):
    """Calculates the displacements of the tiles in microns given resolution, overlaps and image size

    Arguments:
      size (tuple): image size in pixel
      resolution (tuple): image resolution in microns per pixel
      overlap (tuple): overlap of the images in microns
      addOverlapp (tuple): additional overlap in pixel
      units (str): 'Pixel' or 'Microns'

    Returns:
      tuple: displacements in x and y of the tiles
    """

    size = np.array(size, dtype=float)
    size = size[:2]
    resolution = np.array(resolution, dtype=float)
    resolution = resolution[:2]
    overlap = np.array([overlap], dtype=float).flatten()
    overlap = overlap[:2]
    add = np.array([addOverlap], dtype=float).flatten()
    add = np.pad(add, 2, 'wrap')
    add = add[:2]

    if units == 'Pixel':
        overlap_pix = overlap + add
        return (size - overlap_pix)
    else:
        overlap_pix = overlap / resolution + add
        return (size - overlap_pix) * resolution


def xml_import_stack(row, col, zrange, displacement, stitchable=False, directory=None, expression=None, dim=None):
    """Creates a single stack xml information for the xmlf import file of TeraSticher

    Arguments:
      row, col (int or str): the row and column specifications
      zrange (tuple or str): the z range specifications
      displacement (tuple): the (x,y,z) displacements of the stack in pixel / voxel
      stitchable (bool or str):  stitachable parameter
      directory (str): directory of stack
      expression (str or None): regular expression specifing the images of the (row,col) tile
      dim (int or None): dimension of the stack, if None dim = 2

    Returns:
      etree: xml entry as lxml etree
    """

    if dim is None:
        dim = 2

    s = etree.Element('Stack')
    for e in ['NORTH_displacements', 'EAST_displacements', 'SOUTH_displacements', 'WEST_displacements']:
        s.append(etree.Element(e))

    s.set('ROW', str(row))
    s.set('COL', str(col))

    s.set('ABS_H', str(int(displacement[0])))
    s.set('ABS_V', str(int(displacement[1])))
    if len(displacement) > 2:
        s.set('ABS_D', str(int(displacement[2])))
    else:
        s.set('ABS_D', str(0))

    if stitchable:
        s.set('STITCHABLE', 'yes')
    else:
        s.set('STITCHABLE', 'no')

    if directory is None or directory == '':
        directory = '.'
    s.set('DIR_NAME', str(directory))

    if expression is None:
        expression = ''

    s.set('IMG_REGEX', str(expression))

    if dim == 2:
        if isinstance(zrange, tuple) or isinstance(zrange, list):
            zrange = '[%d, %d)' % (zrange[0], zrange[1])
        s.set('Z_RANGES', zrange)

    if dim == 3:
        s.set('N_BLOCKS', str(1))

        if not isinstance(zrange, tuple) and not isinstance(zrange, list):
            raise RuntimeError('zrange needs to be a tuple but found %s', str(zrange))

        s.set('BLOCK_SIZES', str(int(zrange[1]) - int(zrange[0])))
        s.set('BLOCKS_ABS_D', str(0))
        s.set('N_CHANS', str(1))
        s.set('N_BYTESxCHAN', str(2))

    return s


def xml_import(base_directory, size, resolution=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0), overlap=(0.0, 0.0),
               add_overlap=0, tiling=(1, 1), tile_expression=None, z_range=None, rows=None, cols=None, dim=None,
               form=None, xml_import_file=None, as_string=False):
    """Creates an xml import file specifing the data for the TeraStitcher

    Arguments:
      base_directory (str): base directory for the volumetric data
      size (tuple): size of the volume in pixel
      resolution (tuple): resolution of the image n microns per pixel
      origin (tuple): origin of the image
      overlap (tuple): overlap of the tiles in x,y in microns
      add_overlap (tuple): additional overlap to account for in the displacement calculation, usefull to add extra search space
      tiling (tuple): tile dimensions
      tile_expression (function or None): tileExpression(row,col) should return a
                                         regular expression specifing the path and images
                                         of the (row,col) tile, if None all
                                         images in the tile directory are used
      rows,cols (list or None): if list use these values for row and col indices, if None integer ranges are used
      z_range (function or None): zRange(row, col) should return the zrange for
                                 the (row,col) tile as a string, if None, full
                                 range is used
      dim (str, int or None): dimension of a single input tile, if none dim = 2
      form (str, int or None): volume format, if none determine from dim ('TiledXY|2Dseries' or 'TiledXY|3Dseries')
      xml_import_file (str or None): file name to save the xml to, if None return xml
                                 string or xml element tree
      as_string (bool): if True return as string otherwise as xml element tree
    """

    xml = etree.Element('TeraStitcher')

    if dim != 3:
        dim = 2
    if form is None:
        if dim == 2:
            form = 'TiledXY|2Dseries'
        else:
            form = 'TiledXY|3Dseries'

    xml.set('volume_format', form)

    e = etree.Element('stacks_dir')
    e.set('value', base_directory)
    xml.append(e)

    e = etree.Element('voxel_dims')
    e.set('H', str(resolution[0]))
    e.set('V', str(resolution[1]))
    e.set('D', str(resolution[2]))
    xml.append(e)

    e = etree.Element('origin')
    e.set('H', str(origin[0]))
    e.set('V', str(origin[1]))
    e.set('D', str(origin[2]))
    xml.append(e)

    e = etree.Element('mechanical_displacements')
    displ = displacements(size, resolution, overlap, add_overlap, units='Microns')
    e.set('H', str(displ[0]))
    e.set('V', str(displ[1]))
    xml.append(e)

    e = etree.Element('dimensions')
    e.set('stack_columns', str(tiling[1]))
    e.set('stack_rows', str(tiling[0]))
    e.set('stack_slices', str(size[2]))
    xml.append(e)

    if z_range is all or z_range is None:
        z_range = lambda row, col: (0, size[2])
    if isinstance(z_range, tuple) or isinstance(z_range, list):
        zr = z_range
        z_range = lambda row, col: zr

    displ = displacements(size, resolution, overlap, add_overlap, units='Pixel')

    if tile_expression is None:
        tile_expression = lambda row, col: ''

    stack = etree.Element('STACKS')

    if rows is None:
        rows = range(tiling[0])
    if cols is None:
        cols = range(tiling[1])

    for rowi, row in enumerate(rows):
        for coli, col in enumerate(cols):
            stackdispl = [displ[0] * coli, displ[1] * rowi]
            te = tile_expression(row, col)
            dn, fn = os.path.split(te)
            e = xml_import_stack(rowi, coli, z_range(row, col), stackdispl, stitchable=False, directory=dn,
                                 expression=fn,
                                 dim=dim)
            stack.append(e)

    xml.append(stack)

    if xml_import_file is None:
        if as_string:
            return etree.tostring(xml, pretty_print=True, xml_declaration=True, encoding="utf-8",
                                  doctype='<!DOCTYPE TeraStitcher SYSTEM "TeraStitcher.DTD">')
        else:
            return xml
    else:
        xml_string = etree.tostring(xml, pretty_print=True, xml_declaration=True, encoding="utf-8",
                                    doctype='<!DOCTYPE TeraStitcher SYSTEM "TeraStitcher.DTD">')
        f = open(xml_import_file, 'wb')
        f.write(xml_string)
        f.close()
        return xml_import_file


def xml_file_import(regular_expression, size=None, overlap=None, add_overlap=0, origin=None, resolution=None,
                    tiling=None, tile_expression=None, z_range=None, xml_import_file=None, as_string=False, ff=None,
                    manual=False, manual_data=None):
    """Creates the xml import file for TeraStitcher from a directory of image files and optional additional information
    Arguments:
      regularExpression (string): base directory or regular expression to infer the tiled image data, in the latter case use group names
                                  names 'col' and 'row' for the tile indices and 'z' for the z-plane indices
      size (tuple or None): volume size in pixel if None inferred from meta data
      overlap (tuple or None): the (x,y) overlap of the tiles in pixel, if None try to find info from metadata
      addOverlapp (tuple): additional overlap in pixel to increase posssible search radius
      origin (tuple or None): the origin of the image, if None then the tpypical value (0.0,0.0,0.0).
      resolution (tupl0 or None): the (x,y,z) resolution of the image in microns per pixel, if None try to find info from metadata
      tiling (tuple or  None): the (row,col) tiling dimensions, if None infer from 'row' and 'col' groups in regularExpression
      tileExpression (function or None): a function of (col,row) that returns a regular expression for the images in the (col,row) tile or None if that tile is missing,
                                         if None use the regular expression row and col group info
      zRange (function or None. all): a function of (col,row) that returns a the z range specifications for the (col,row) tile
                                      if None use the regular expression z group to infer this from first tile, if all infer this in detail for all tiles
      xmlImportFile (string or None): filename for the xml import file, if None return the xml tree, if all create 'TreaStitcher_import.xml' file in base directory
      asString (bool): if True return xml code as string, otherwise return as a xml tree

    Returns:
      string: filename of the xml import file

    Note:
      Example for a regular expression: r'/path/to/image/image_(?P<row>\d{2})_(?P<col>\d{2})_(?P<z>\d{4}).tif'

    See also:
      :param manual:
      :func:`importData`
    """

    ## origin
    if origin is None:
        origin = (0, 0, 0)

    if manual:
        size = manual_data[0]
        resolution = manual_data[1]
        overlap = manual_data[2]

    if ff != None:

        first_file = find_first_file(ff)
        finfo = find_file_info(first_file)

        if size is None:
            size = finfo['size']

        if resolution is None:
            resolution = finfo['resolution']

        if overlap is None:
            overlap = finfo['overlap']

        for val, name in zip([size, overlap, resolution], ['size', 'overlap', 'resolution']):
            if val is None:
                raise RuntimeError('cannot determine %s from file or input!' % name)

    # infer size, resolution and overlap
    if ff == None:

        if size is None or resolution is None or overlap is None:
            first_file = find_first_file(regular_expression)

            finfo = find_file_info(first_file)

            if size is None:
                size = finfo['size']

            if resolution is None:
                resolution = finfo['resolution']

            if overlap is None:
                overlap = finfo['overlap']

            for val, name in zip([size, overlap, resolution], ['size', 'overlap', 'resolution']):
                if val is None:
                    raise RuntimeError('cannot determine %s from file or input!' % name)

    if tiling is None or z_range is None or z_range is all:
        # infer tiling from regular expression
        # find file:
        fns, ids = find_file_list(regular_expression, groups=('row', 'col', 'z'))
        print(fns, ids)
        fsize = io.data_size(fns[0])
        dim = len(fsize)

        ids = np.array(ids)

        # get rid of invalid labels
        b = np.zeros(ids.shape[0], dtype=bool)
        for i in range(5 - dim):
            b = np.logical_or(b, np.equal(ids[:, i], None))
        b = np.logical_not(b)
        fns = fns[b]
        ids = ids[b]

        if len(fns) == 0:
            raise RuntimeError(
                'no files found that match the expression %s with row, col and z groups' % regular_expression)

        # calculate tile dimensions
        rows = np.unique(ids[:, 0])
        nrows = len(rows)

        cols = np.unique(ids[:, 1])
        ncols = len(cols)

        if tiling is not None and tiling != (nrows, ncols):
            raise RuntimeWarning('specified tiling is different from inferred tiling, min tile number will be used !')
            tiling = (min(nrows, tiling[0]), min(ncols, tiling[1]))
            rows = rows[:tiling[0]]
            cols = cols[:tiling[1]]
        else:
            tiling = (nrows, ncols)

        # zRanges
        if dim == 2:
            zs = np.unique(ids[:, 2])
            nzs = len(zs)

            if z_range is None:
                z_range = (0, nzs)
            elif z_range is all:
                z_range = lambda row, col: (0, np.sum(np.logical_and(ids[:, 0] == row, ids[:, 1] == col)))
        else:
            nzs = fsize[2]
            z_range = (0, nzs)

    else:
        rows = None
        cols = None
        fns = None

        nzs = 0
        for row in range(tiling[0]):
            for col in range(tiling[1]):
                nzs = max(nzs, z_range(row, col)[1])

    size = tuple(size) + (nzs,)

    # base directory and tile directories
    if fns is None:
        if first_file is None:
            fn = find_first_file(regular_expression, sort=False)
        else:
            fn = first_file
    else:
        fn = fns[0]

    fdim = len(io.data_size(fn))

    fnsep = fn.split(os.path.sep)
    fesep = regular_expression.split(os.path.sep)

    if len(fnsep) != len(fesep):
        raise RuntimeError('inconsistent file names and file expression!')

    for i in range(len(fnsep)):
        if fnsep[i] != fesep[i]:
            base_directory = os.path.sep.join(fesep[:i])
            regular_expression = os.path.sep.join(fesep[i:])
            break

    # tileExpression
    if tile_expression is None:
        def makeTileExpression(row, col):
            te = re.sub(r"\(\?P<row>.*?\)", str(row), regular_expression, count=1)
            return re.sub(r"\(\?P<col>.*?\)", str(col), te, count=1)

        tile_expression = makeTileExpression
    elif isinstance(tile_expression, str):
        tileExpressionString = tile_expression

        def makeTileExpression(row, col):
            te = re.sub(r"\(\?P<row>.*?\)", str(row), tileExpressionString, count=1)
            return re.sub(r'\(\?P<col>.*?\)', str(col), te, count=1)

        tile_expression = makeTileExpression

    # create xml import
    return xml_import(base_directory, size=size, resolution=resolution, origin=origin, overlap=overlap,
                      add_overlap=add_overlap, tiling=tiling, tile_expression=tile_expression, z_range=z_range,
                      rows=rows, cols=cols, dim=fdim, xml_import_file=xml_import_file, as_string=as_string)


def base_directory_from_xml_import(xmlFile):
    """Extracts the base directory specified in the xml descriptor

    Arguments:
      xmlFile: the TeraStitcher xml file

    Returns:
      str: base directory specified int he xml file
    """

    try:
        xmlt = etree.parse(xmlFile)
        base_directory = xmlt.find('stacks_dir').get('value')
        return base_directory
    except:
        return None


def delete_bin_file(base_directory):
    """Clears the base directory from generated binary file

    Arguments:
      base_directory (str): the base directory or xml descriptor file

    Note:
      TeraStitcher produces a 'mdata.bin' file during import
    """

    bD = base_directory_from_xml_import(base_directory)
    if bD is None:
        bD = base_directory

    try:
        bin_dile = os.path.join(bD, 'mdata.bin')
        os.remove(bin_dile)
    except:
        pass


def import_data(xml_import_file=None, base_directory=None, resolution=None, orientation=None,
                regular_expression=None, form=None, rescan=None, sparse=None, xml_result_file=None, clear=True,
                clear_bin=True, verbose=False):
    """Runs the import commmand of TeraSticher generating the xml import file from a folder of files

    Arguments:
      xml_import_file (str or None): the xml import descriptor
      base_directory (str or None): the base directory of the image data, None if xmlImportFile is given
      regular_expression (str or None): optional regular expression for images
      resolution (tuple or None): optional resolution of the image in micorns per pixel
      orientation (tuple or None): optional orientation of the image
      form (str or None): the import format (if None the default 'TiledXY|2Dseries' is used)
      rescan (bool or None): optional rescan of image data
      sparse (bool or None): optional switch to specify if data is sparse
      xmlResultFileName (str or None): output xml file name
      clear (bool): if True delete previous imported binary files
      clear_bin (bool): if True remove the mdata.bin file that can cause problems in downstream processing

    Returns:
      str: xml import file name

    See also:
      :func:`xmlImportFile`, `TeraStitcher import step <https://github.com/abria/TeraStitcher/wiki/Step-1:-Import>`_.
    """

    check_sticher_initialized()
    global terastitcher_binary

    if base_directory is None and xml_import_file is None:
        raise RuntimeError('importData requires baseDirectory or xmlImportFile!')

    if xml_import_file is None:
        if resolution is None:
            resolution = (1.0, 1.0, 1.0)
        if orientation is None:
            orientation = (1, 2, 3)

    cmd = terastitcher_binary + ' --import '

    if base_directory is not None:
        cmd = cmd + ' --volin="' + base_directory + '" '

    if xml_import_file is not None:
        cmd = cmd + ' --projin="' + xml_import_file + '" '

    # voxel size / resolution
    if resolution is not None:
        vsize = ['--vxl1=', '--vxl2=', '--vxl3=']
        for i in range(3):
            cmd = cmd + vsize[i] + str(resolution[i]) + ' '

    # reference orientation1
    if orientation is not None:
        ref = ['--ref1=', '--ref2=', '--ref3=']
        # refnames = ['x', 'y', 'z']
        for i in range(3):
            # if orientation[i] < 0:
            #  rn = '-'
            # else:
            #  rn = ''
            # rn = rn + refnames[orientation[abs(i)]]
            rn = str(orientation[i])
            cmd = cmd + ref[i] + rn + ' '

    # optional arguments
    if xml_result_file is not None:
        cmd = cmd + '--projout="' + xml_result_file + '" '
    else:
        if xml_import_file is not None:
            xml_result_file = xml_import_file
        else:
            xml_result_file = 'xml_import.xml'

    if form is not None:
        cmd = cmd + '--volin_plugin="' + form + '" '
    # else:
    #  cmd = cmd + '--volin_plugin="TiledXY|2Dseries" '

    if rescan is True:
        cmd = cmd + '--rescan '

    if sparse is True:
        cmd = cmd + '--sparse_data '

    if regular_expression is not None:
        cmd = cmd + '--imin_regex="' + regular_expression + '" '

    if clear or clear_bin:
        delete_bin_file(xml_import_file)

    res = os.system(cmd)

    if res != 0:
        raise RuntimeError('import_data: failed executing: ' + cmd)

    if verbose:
        ut.print_c("[INFO] XML data import: complete!")

    if clear_bin:
        delete_bin_file(base_directory)

    return xml_result_file


def align_data(xml_import_file: object, slices: object = None, sub_region: object = None, overlap: object = None,
               search: object = None, algorithm: object = None,
               channel: object = None,
               xml_result_file: object = None,
               clear_bin: object = True, silent_mode: object = True, verbose: object = False) -> object:
    """Runs the alignment commmand of TeraSticher aligning the tiles

    Arguments:
      xml_import_file (str or None): the xml import descriptor
      slices (int or None): optional number of top slices to use for alignment
      sub_region (tuple or None): optional sub region in the form ((xmin,xmax),(ymin,ymax), (zmin, zmax))
      overlap (tuple or None): optional estimated overlap
      search (tuple or None): optional search region as pixel radii, i.e. (xradius, yradius)
      algorithm (str or None): optional algorithm 'MIPNCC' for MIP Normalized Cross-Correlation or 'PC' for Phase Correlation
      channel (str or None): optional channel to use 'R', 'G', 'B' or 'all'
      xml_result_file (str or None): output xml displacement file name
      clear_bin (bool): remove mdata.bin file to avoid errors

    Returns:
      str : the result xml displacement file name

    See also:
      `TeraStitcher align step <https://github.com/abria/TeraStitcher/wiki/Step-2:-Align>`_.
    """

    check_sticher_initialized()
    global terastitcher_binary

    if clear_bin:
        delete_bin_file(xml_import_file)

    cmd = terastitcher_binary + ' --displcompute '

    cmd = cmd + ' --projin="' + xml_import_file + '" '

    # slices
    if slices is not None:
        cmd = cmd + ' --subvoldim=' + str(slices) + ' '

    if sub_region is not None:
        sns = (('--R0=', '--R1='), ('--C0=', '--C1='), ('--D0=', '--D1='))
        for d in range(3):
            for m in range(2):
                if sub_region[d][m] is not None and sub_region[d][m] is not all:
                    cmd = cmd + sns[d][m] + str(sub_region[d][m]) + ' '

    if overlap is not None:
        ov = ('--oH=', '--oV=')
        for d in range(2):
            if overlap[d] is not None:
                cmd = cmd + ov[d] + str(overlap[d]) + ' '

    if search is not None:
        sd = ('--sH=', '--sV=', '--sD=')
        for d in range(3):
            if search[d] is not None:
                cmd = cmd + sd[d] + str(search[d]) + ' '

    if algorithm is not None:
        cmd = cmd + '--algorithm="' + algorithm + '" '

    if channel is not None:
        cmd = cmd + '--imin_channel="' + channel + '" '

    if xml_result_file is not None:
        cmd = cmd + '--projout="' + xml_result_file + '" '
    else:
        xml_result_file = 'xml_displcomp.xml'

    if verbose:
        if silent_mode:
            cmd = cmd + ("2>&1 | "
                         "grep --line-buffered -E 'PROGRESS:	[0-9]+%' | "
                         "awk '{ printf \"\\r%s\", $0 fflush() } END { print \"\" }'")

        ut.print_c("[INFO] Running pairwise displacement computation")
    else:
        cmd = cmd + (" > /dev/null 2>&1")
    res = os.system(cmd)

    if res != 0:
        raise RuntimeError('alignData: failed executing: ' + cmd)

    if verbose:
        ut.print_c("[INFO] Pairwise displacement computation: complete!")

    return xml_result_file


def project_displacements(xml_displacement_file, xml_result_file=None, delete_bin=True, verbose=False):
    """Runs the projection step of TeraSticher to aligning the tiles

    Arguments:
      xml_displacement_file (str or None): the xml displacement descriptor
      xml_result_file (str or None): output xml projection file name
      delete_bin (bool): delete binary file generated by TeraSticher import before projecting data

    Returns:
      str : the result xml projection file name

    See also:
      `TeraStitcher project step <https://github.com/abria/TeraStitcher/wiki/Step-3:-Project>`_.

    Note:
       Use deleteBin = True option to prevent conflicts/errors with the binary data file generated by import command.
    """

    check_sticher_initialized()
    global terastitcher_binary

    if delete_bin:
        delete_bin_file(xml_displacement_file)

    cmd = terastitcher_binary + ' --displproj '

    cmd = cmd + ' --projin="' + xml_displacement_file + '" '

    if xml_result_file is not None:
        cmd = cmd + '--projout="' + xml_result_file + '" '
    else:
        xml_result_file = 'xml_displproj.xml'

    res = os.system(cmd)

    if res != 0:
        raise RuntimeError('project_displacements: failed executing: ' + cmd)

    if verbose:
        ut.print_c("[INFO] Project displacements: complete!")

    return xml_result_file


def threshold_displacements(xml_projection_file, threshold=None, xml_result_file=None, verbose=False):
    """Runs the thresholding step of TeraSticher to aligning the tiles
  
    Arguments:
      xml_projection_file (str or None): the xml projection descriptor
      threshold (float or None): optional threshold value
      xml_result_file (str or None): output xml thresholded file name
  
    Returns:
      str : the result xml thresholded file name
  
    See also:
      `TeraStitcher project step <https://github.com/abria/TeraStitcher/wiki/Step-4:-Threshold>`_.
    """

    check_sticher_initialized()
    global terastitcher_binary

    cmd = terastitcher_binary + ' --displthres '

    cmd = cmd + ' --projin="' + xml_projection_file + '" '

    if threshold is None:
        threshold = 0
    cmd = cmd + ' --threshold=' + str(threshold) + ' '

    if xml_result_file is not None:
        cmd = cmd + '--projout="' + xml_result_file + '" '
    else:
        xml_result_file = 'xml_displthres.xml'

    res = os.system(cmd)

    if res != 0:
        raise RuntimeError('threshold_displacements: failed executing: ' + cmd)

    if verbose:
        ut.print_c("[INFO] Threshold displacements: complete!")

    return xml_result_file


def place_tiles(xml_threshold_file, algorithm=None, xml_result_file=None, verbose=False):
    """Runs the placement step of TeraSticher to aligning the tiles
  
    Arguments:
      xmlThresholdFile (str or None): the xml threshold descriptor
      algorithm (str or None): optional algorithm to use for placement: 
                               'MST' for Minimum Spanning Tree
                               'SCAN_V' for scanning along vertical axis
                               'SCAN_H' for scannning along horizontal axis
      xmlResultFile (str or None): output xml placed file name
  
    Returns:
      str : the result xml placed file name
  
    See also:
      `TeraStitcher project step <https://github.com/abria/TeraStitcher/wiki/Step-5:-Place>`_.
    """'--oV='

    check_sticher_initialized()
    global terastitcher_binary

    cmd = terastitcher_binary + ' --placetiles '

    cmd = cmd + ' --projin="' + xml_threshold_file + '" '

    if algorithm is not None:
        cmd = cmd + ' --algorithm="' + algorithm + '" '

    if xml_result_file is not None:
        cmd = cmd + '--projout="' + xml_result_file + '" '
    else:
        xml_result_file = 'xml_merging.xml'

    res = os.system(cmd)

    if res != 0:
        raise RuntimeError('place_tiles: failed executing: ' + cmd)

    if verbose:
        ut.print_c("[INFO] Tile placements: complete!")

    return xml_result_file


def stitch_data(xml_placement_file, result_path, algorithm=None, resolutions=None, form=None, channel=None,
                sub_region=None,
                bit_depth=None, block_size=None, compress=False, silent_mode=True):
    """Runs the final stiching step of TeraSticher

    Arguments:
      xml_placement_file (str or None): the xml placement descriptor
      result_path (str): result path, file name or file expression for the stiched data
      algorithm (str or None): optional algorithm to use for placement:
                               'NOBLEND' for no blending
                               'SINBLEND' for sinusoidal blending
      resolutions (tuple or None): the different resolutions to produce
      form (str or None): the output form, if None determined automatically
      channel (str or None): the channels to use, 'R', 'G', 'B' or 'all'
      sub_region (tuple or None): optional sub region in the form ((xmin,xmax),(ymin,ymax), (zmin, zmax))
      bit_depth (int or None): the pits per pixel to use, default is 8
      block_size (tuple): the sizes of various blocks to save stiched image into
      cleanup (bool): if True delete the TeraSticher file structure
      compress (bool): if True compress final tif images

    Returns:
      str : the result path or file name of the stiched data

    See also:
      `TeraStitcher project step <https://github.com/abria/TeraStitcher/wiki/Step-6:-Merge>`_.
    """

    check_sticher_initialized()
    global terastitcher_binary

    cmd = terastitcher_binary + ' --merge --imout_format="tif" '

    cmd = cmd + '--projin="' + xml_placement_file + '" '

    if len(result_path) > 3 and result_path[-4:] == '.tif':
        if fu.is_file_expression(result_path, check=False):
            form = 'TiledXY|2Dseries'
            result_path, filename = os.path.split(result_path)
        else:
            form = 'TiledXY|3Dseries'
            result_path, filename = os.path.split(result_path)
    else:
        filename = None

    cmd = cmd + '--volout="' + result_path + '" '

    if algorithm is not None:
        cmd = cmd + ' --algorithm="' + algorithm + '" '

    if resolutions is not None:
        cmd = cmd + '--resolutions="'
        for r in sorted(resolutions):
            cmd = cmd + str(r)
        cmd = cmd + ' '

    if form is not None:
        cmd = cmd + '--volout_plugin="' + form + '" '

    if channel is not None:
        cmd = cmd + '--imin_channel="' + channel + '" '

    if sub_region is not None:
        sns = (('--R0=', '--R1='), ('--C0=', '--C1='), ('--D0=', '--D1-'))
        for d in range(3):
            for m in range(2):
                if sub_region[d][m] is not None:
                    cmd = cmd + sns[d][m] + str(sub_region[d][m]) + ' '

    if block_size is not None:
        bs = ('--slicewidth=', '--sliceheight=', '--slicedepth=')
        for d in range(3):
            if block_size[d] is not None:
                cmd = cmd + bs[d] + str(block_size[d]) + ' '

    if bit_depth is not None:
        cmd = cmd + '--imout_depth=' + str(bit_depth) + ' '

    if not compress:
        cmd = cmd + '--libtiff_uncompress '

        # print resultPath
    fu.create_directory(result_path, split=False)

    if silent_mode:
        cmd = cmd + ("2>&1 | grep --line-buffered -E 'PROGRESS:	[0-9]+%' | "
                     "awk '{ printf \"\\r%s\", $0 fflush() } END { print \"\" }'")

    res = os.system(cmd)

    if res != 0:
        raise RuntimeError('stitchData: failed executing: ' + cmd)

    ut.print_c("[INFO] Stitching: complete!")

    if filename is not None:
        if fu.is_file_expression(filename, check=False):  # convert list of files in TeraSticher from
            # TODO: multiple resolutions
            basedir = max(glob.glob(os.path.join(result_path, '*')), key=os.path.getmtime)
            move_tera_stitcher_stack_to_file_list(basedir, os.path.join(result_path, filename),
                                                  delete_directory=True, verbose=False)

        else:  # single file in TeraSticher folder
            # get most recent created file
            # TODO: test if this works
            img_file = max(glob.glob(os.path.join(result_path, '*/*/*/*')), key=os.path.getmtime)
            filename = os.path.join(result_path, filename)
            os.rename(img_file, filename)
            img_path = os.path.sep.join(img_file.split(os.path.sep)[:-3])
            shutil.rmtree(img_path)
            return filename

    else:
        return result_path


def move_tera_stitcher_stack_to_file_list(source, sink, delete_directory=True, verbose=True):
    """Moves image files from TeraSticher file structure to a list of files

    Arguments:
      source (str): base directory of the TeraStitcher files
      sink (str): regular expression of the files to copy to
      verbose (bool): show progress
    Returns:
      str: sink regular expression
    """

    fns = glob.glob(os.path.join(source, '*/*/*'))
    fns = natsorted(fns)

    fu.create_directory(sink)
    for i, f in enumerate(fns):
        fn = fl.file_expression_to_file_name(sink, i)
        if verbose:
            print('%s -> %s' % (f, fn))
        shutil.move(f, fn)

    if delete_directory:
        p, _ = os.path.split(fns[0])
        p = p.split(os.path.sep)
        p = p[:-2]
        p = os.path.sep.join(p)
        shutil.rmtree(p)

    return sink


def stitch_samples(raw_directory, **kwargs):
    if not kwargs["study_params"]["scanning_system"] == "bruker":
        if len(os.listdir(raw_directory)) == 0:
            ut.CmliteError(f"No samples were found in: {raw_directory}")

        sample_names = ut.get_sample_names(raw_directory, **kwargs)

        for sample_name in sample_names:
            sample_directory = os.path.join(raw_directory, sample_name)
            stitch_sample(sample_directory, **kwargs)
    else:
        ut.print_c(f"[WARNING] Scans were performed with Bruker SPIM, the data is already stitched!")


def stitch_sample(sample_directory, **kwargs):
    print("")
    sample_name = os.path.basename(sample_directory)
    for channel_to_stitch in kwargs["study_params"]["channels_to_stitch"]:
        processed_folder = os.path.join(sample_directory, f"processed_tiles_{channel_to_stitch}")
        processed_files_expression = "_".join([i for i in os.listdir(processed_folder) if
                                               i.endswith(".tif")][0].split("_")[:-2])
        stitched_folder = os.path.join(sample_directory, f"stitched_{channel_to_stitch}")
        if os.path.exists(processed_folder) and not os.path.exists(stitched_folder):
            ut.print_c(f"[INFO {sample_name}] Channel {channel_to_stitch}: Starting stitching!")
            processed_directory = os.path.join(sample_directory, f"processed_tiles_{channel_to_stitch}")
            expression_raw = rf"{processed_files_expression}_\[(?P<row>\d{{2}})\ x (?P<col>\d{{2}})\]_{channel_to_stitch}.tif"

            metadata_file = os.path.join(sample_directory, "scan_metadata.json")
            if not os.path.exists(metadata_file):
                raise ut.CmliteError("Metadata file is missing!")
            else:
                with open(metadata_file, 'r') as json_file:
                    metadata = json.load(json_file)

            tile_size = np.array([metadata["tile_x"], metadata["tile_y"]])
            resolution = np.array([metadata["x_res"], metadata["y_res"], metadata["z_res"]])
            overlap = tile_size * (metadata["overlap"])

            import_file = xml_file_import(os.path.join(processed_directory, expression_raw),
                                          manual=True,
                                          manual_data=[tile_size, resolution, overlap],
                                          xml_import_file=os.path.join(processed_directory,
                                                                       'terastitcher_import.xml'))

            import_data(import_file, verbose=False)

            mdata_file = os.path.join(processed_directory, 'mdata.bin')
            if os.path.exists(mdata_file):
                os.remove(mdata_file)

            align_file = align_data(import_file,
                                    overlap=overlap.astype(int),
                                    search=kwargs["stitching"]["search_params"],
                                    xml_result_file=os.path.join(processed_directory, 'terastitcher_align.xml'),
                                    sub_region=((all, all), (all, all), kwargs["stitching"]["z_subreg_alignment"]),
                                    algorithm="MIPNCC",
                                    verbose=False)

            project_file = project_displacements(align_file,
                                                 xml_result_file=os.path.join(processed_directory,
                                                                              'terastitcher_project.xml'),
                                                 verbose=False)

            thresh_file = threshold_displacements(project_file,
                                                  xml_result_file=os.path.join(processed_directory,
                                                                               'TeraStitcher_threshold.xml'),
                                                  verbose=False)

            place_file = place_tiles(thresh_file,
                                     xml_result_file=os.path.join(processed_directory, 'TeraStitcher_place.xml'),
                                     verbose=False)

            result = stitch_data(place_file, result_path=os.path.join(stitched_folder, 'Z\d{4}.tif'),
                                 bit_depth=16, algorithm='SINBLEND')

        else:
            ut.print_c(f"[WARNING {sample_name}] Stitching skipped for channel {channel_to_stitch}: "
                       f"stitched_{channel_to_stitch} folder already exists!")
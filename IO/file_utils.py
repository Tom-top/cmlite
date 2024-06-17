import os
import re
import shutil
import importlib

from natsort import natsorted

import utils.regular_expression as cre


def is_directory(dirname):
    """
    Checks if a directory exists.

    Arguments
    ---------
    dirname : str
        The directory name.

    Returns
    -------
    is_directory : bool
        True if source is a real file.
    """
    if not isinstance(dirname, str):
        return False

    return os.path.isdir(dirname)


def is_file(source):
    """Checks if source is an existing file, returns false if it is a directory

    Arguments:
        source (str): source file name

    Returns:
        bool: true if source is a real file
    """

    if not isinstance(source, str):
        return False

    if os.path.exists(source):
        if os.path.isdir(source):
            return False
        else:
            return True
    else:
        return False


def is_file_expression(source, check=True):
    """Checks if source is an exsiting file expression with at least one file matching

    Arguments:
        source (str): source file name

    Returns:
        bool: true if source is a real file matching the file expression
    """

    if not isinstance(source, str):
        return False

    if not cre.is_expression(source, nPatterns=0, exclude=['any']):
        return False

    if is_file(source):
        return False

    if check:
        try:
            fp, fl = read_file_list(source, sort=False)
        except:
            return False
        return len(fl) > 0
    else:
        return True


def read_file_list(filename, sort=True):
    """Returns the list of files that match the regular expression

    Arguments:
        filename (str): file name as regular expression
        sort (bool): sort files naturally

    Returns:
        str, list: path of files, file names that match the regular expression
    """

    # get path
    (fpath, fname) = os.path.split(filename)
    fnames = os.listdir(fpath)

    searchRegex = re.compile(fname).search
    fl = [l for l in fnames for m in (searchRegex(l),) if m]

    if fl == []:
        raise RuntimeError('No files found in ' + fpath + ' matching ' + fname + ' !')
    if sort:
        fl = natsorted(fl)

    return fpath, fl


def file_extension(filename):
    """Returns file extension if exists

    Arguments:
        filename (str): file name

    Returns:
        str: file extension or None
    """

    if not isinstance(filename, str):
        return None

    fext = filename.split('.')
    if len(fext) < 2:
        return None
    else:
        return fext[-1]


def join(path, filename):
    """
    Joins a path to a file name.

    Arguments
    ---------
    path : str
        The path to append a file name to.
    filename : str
        The file name.

    Returns
    -------
    filename : str
        The full file name.
    """
    # TODO: correct to allow joining '/foo' with '/bar' to /foo/bar (os gives /bar!)
    if len(filename) > 0 and filename[0] == '/':
        filename = filename[1:]

    return os.path.join(path, filename)


def split(filename):
    """
    Splits a file name into it's path and name.

    Arguments
    ---------
    filename : str
        The file name.

    Returns
    -------
    path : str
        The path of the file.
    filename : str
        The file name.
    """
    return os.path.split(filename)


def abspath(filename):
    """
    Returns the filename using the full path specification.

    Arguments
    ---------
    filename : str
        The file name.

    Returns
    -------
    filename : str
        The full file name.
    """
    return os.path.abspath(filename)


def create_directory(filename, create=True, split=True):
    """Creates the directory of the file if it does not exists

    Arguments:
        filename (str): file name

    Returns:
        str: directory name
    """
    if split:
        dirname, fname = os.path.split(filename)
    else:
        dirname = filename

    if create and not os.path.exists(dirname):
        os.makedirs(dirname)

    return dirname


def delete_file(filename):
    """
    Deletes a file.

    Arguments
    ---------
    filename : str
        Filename to delete.
    """
    if is_file(filename):
        os.remove(filename)


def copy_file(source, sink):
    """Copy a file.

    Arguments
    ---------
    source : str
        Filename of the file to copy.
    sink : str
        File or directory name to copy the file to.

    Returns
    -------
    sink : str
        The name of the copied file.
    """
    if is_directory(sink):
        path, name = os.path.split(source)
        sink = os.path.join(sink, name)
    shutil.copy(source, sink)
    return sink


def uncompress(file_path, extension='zip', check=True, verbose=False):
    """
    Unzips a file only if 1) the file does not exist (check), 2) the compressed file exists.

    Arguments
    ---------
    file_path : str
        The file path to search for.
    extension : str
        The extension for the compressed file.
    check : bool
        If True, check if the decompressed file already exists.
    verbose : bool
        Print progress info.

    Returns
    -------
    filename : str or None
        The uncompressed filename or None if failed.
    """
    if not os.path.exists(file_path) or not check:
        if extension == 'auto':
            for algo in ('zip', 'bz2', 'gzip', 'lzma'):
                f_path_w_ext = f'{file_path}.{algo}'
                print(f_path_w_ext)
                if os.path.exists(f_path_w_ext):
                    extension = algo
                    break
            else:
                raise ValueError(f'Could not find compressed source for {file_path}')

        compressed_path = f'{file_path}.{extension}'
        if os.path.exists(compressed_path):
            if verbose:
                print(f'Decompressing source: {compressed_path}')
            if extension == 'zip':
                import zipfile
                try:
                    with zipfile.ZipFile(compressed_path, 'r') as zipf:
                        if os.path.splitext(file_path)[-1] in ('.tif', '.nrrd'):
                            zipf.extract(os.path.basename(file_path), path=os.path.dirname(compressed_path))
                        else:
                            zipf.extractall(path=os.path.dirname(compressed_path))
                    if not os.path.exists(file_path):
                        raise FileNotFoundError
                except Exception as err:  # FIXME: TOO broad
                    print(err)
                    return
            elif extension in ('bz2', 'gzip', 'lzma'):
                mod = importlib.import_module(extension)
                with open(file_path, 'wb') as out, \
                        open(compressed_path, 'rb') as compressed_file:
                    out.write(mod.decompress(compressed_file.read()))
            else:
                raise NotImplementedError(f'Unrecognized compression extension {extension}')
        else:
            print(f'Cannot find compressed source: {compressed_path}')
    return file_path
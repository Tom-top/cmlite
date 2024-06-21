import sys

self = sys.modules[__name__]

import numpy as np

import IO.IO as io


def read_data_group(filenames, combine=True, **args):
    """Turn a list of filenames for data into a numpy stack"""

    # check if stack already:
    if isinstance(filenames, np.ndarray):
        return filenames

    # read the individual files
    group = []
    for f in filenames:
        data = io.read(f, **args)
        data = np.reshape(data, (1,) + data.shape)
        group.append(data)

    if combine:
        return np.vstack(group)
    else:
        return group
import multiprocessing as mp

import parallel_processing.shared_memory_array as sma

__all__ = ['get', 'insert', 'free', 'clean', 'zeros']


###############################################################################
### Manager
###############################################################################

class SharedMemmoryManager(object):
    """SharedMemmoryManager provides handles to shared arrays for parallel processing."""

    _instance = None
    """Pointer to global instance"""

    __slots__ = ['arrays', 'current', 'count', 'lock']

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(SharedMemmoryManager, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        self.arrays = [None] * 32
        self.current = 0
        self.count = 0
        self.lock = mp.Lock()

    def handle(self):
        # double size if necessary
        if (self.count >= len(self.arrays)):
            self.arrays = self.arrays + [None] * len(self.arrays)
        # find free handle
        while self.arrays[self.current] is not None:
            self.current = (self.current + 1) % len(self.arrays)
        return self.current

    @staticmethod
    def instance():
        if not SharedMemmoryManager._instance:
            SharedMemmoryManager._instance = SharedMemmoryManager()
        return SharedMemmoryManager._instance

    @staticmethod
    def zeros(shape, dtype=None, order=None):
        self = SharedMemmoryManager.instance()
        self.lock.acquire()
        # next handle
        self.handle()
        # create array in shared memory segment and wrap with numpy
        self.arrays[self.current] = sma.zeros(shape, dtype, order)
        # update cnt
        self.count += 1
        self.lock.release()
        return self.current

    @staticmethod
    def insert(array):
        self = SharedMemmoryManager.instance()
        # next handle
        self.handle()
        # convert to shared array and insert into handle
        self.arrays[self.current] = sma.as_shared(array)
        # update cnt
        self.count += 1
        return self.current

    @staticmethod
    def free(hdl):
        self = SharedMemmoryManager.instance()
        self.lock.acquire()
        # set reference to None
        if self.arrays[hdl] is not None:  # consider multiple calls to free
            self.arrays[hdl] = None
            self.count -= 1
        self.lock.release()

    @staticmethod
    def clean():
        self = SharedMemmoryManager.instance()
        self.lock.acquire()
        for i in range(len(self.arrays)):
            self.arrays[i] = None
        self.current = 0
        self.count = 0
        self.lock.release()

    @staticmethod
    def get(i):
        self = SharedMemmoryManager.instance()
        return self.arrays[i]


###############################################################################
### Functionality
###############################################################################

def zeros(shape, dtype=None, order=None):
    """Creates a shared zero array and inserts it into the shared memory manager.

    Arguments
    ---------
    shape : tuple
      Shape of the array.
    dtype : dtype or None
      The type of the array.
    order : 'C', 'F', or None
      The contiguous order of the array.

    Returns
    -------
    handle : int
      The handle to this array.
    """
    return SharedMemmoryManager.zeros(shape=shape, dtype=dtype, order=order)


def get(handle):
    """Returns the array in the shared memory manager with given handle.

    Arguments
    ---------
    handle : int
      Shared memory handle of the array.

    Returns
    -------
    array : array
      The shared array with the specified handle.
    """
    return SharedMemmoryManager.get(handle)


def insert(array):
    """Inserts the array in the shared memory manager.

    Arguments
    ---------
    array : array
      The array to insert into the shared memory manager.

    Returns
    -------
    handle : int
      The shared array handle.
    """
    return SharedMemmoryManager.insert(array)


def free(handle):
    """Removes the array with given handle from the shared memory manager.

    Arguments
    ---------
    handle : int
      Shared memory handle of the array.
    """
    SharedMemmoryManager.free(handle)


def clean():
    """Removes all references to the shared arrays."""
    SharedMemmoryManager.clean()
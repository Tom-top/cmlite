#cython: language_level=3, boundscheck=False, wraparound=False, nonecheck=False, initializedcheck=False, cdivision=True

import numpy as np
cimport numpy as np

cimport cython
from cython.parallel import prange, parallel

ctypedef fused point_t:
  np.int32_t
  np.int64_t
  np.uint8_t
  np.uint16_t
  np.uint32_t
  np.uint64_t
  np.float32_t 
  np.float64_t

ctypedef fused sink_t:
  np.int32_t
  np.int64_t
  np.uint8_t
  np.uint16_t
  np.uint32_t
  np.uint64_t
  np.float32_t 
  np.float64_t

ctypedef fused weight_t:
  np.int32_t
  np.int64_t
  np.uint8_t
  np.uint16_t
  np.uint32_t
  np.uint64_t
  np.float32_t 
  np.float64_t

ctypedef np.float64_t kernel_t

ctypedef Py_ssize_t index_t

cdef extern from "stdio.h":
  int printf(char *format, ...) nogil

###############################################################################
### Devolution 
###############################################################################

#@cython.boundscheck(False)
#@cython.wraparound(False)
cpdef void devolve_uniform(point_t[:,:] points, index_t[:,:] indices, sink_t[:] sink, index_t[:] shape, index_t[:] strides, int processes) nogil:
  """Converts a list of points into an volumetric image array."""
  
  cdef index_t i, j, k, d, n, v
  cdef index_t n_points  = points.shape[0];
  cdef index_t n_sink    = sink.shape[0];
  cdef index_t n_indices = indices.shape[0];                                   
  cdef index_t n_dim     = strides.shape[0];
  
  with nogil, parallel(num_threads = processes):    
    for n in prange(n_points, schedule='guided'):
      for i in range(n_indices):
        j = 0;
        v = 1;
        for d in range(n_dim):
          k = <index_t>points[n,d] + indices[i,d];
          if not (0 <= k and k < shape[d]):
            v = 0;
            break;
          else:
            j = j + k * strides[d];
        if v == 1:
          sink[j] += 1;
    
  return;


#@cython.boundscheck(False)
#@cython.wraparound(False)
cpdef void devolve_uniform_kernel(point_t[:,:] points, index_t[:,:] indices, kernel_t[:] kernel, sink_t[:] sink, index_t[:] shape, index_t[:] strides, int processes) nogil:
  """Converts a list of points into an volumetric image array."""
  
  cdef index_t i, j, k, v, d, n
  cdef index_t n_points  = points.shape[0];
  cdef index_t n_sink    = sink.shape[0];
  cdef index_t n_indices = indices.shape[0];                                   
  cdef index_t n_dim     = strides.shape[0];
  
  with nogil, parallel(num_threads = processes):    
    for n in prange(n_points, schedule='guided'): 
      for i in range(n_indices):
        j = 0;
        v = 1;
        for d in range(n_dim):
          k = <index_t>points[n,d] + indices[i,d];
          if not (0 <= k and k < shape[d]):
            v = 0;
            break;
          else:
            j = j + k * strides[d];
        if v == 1:
          sink[j] += <sink_t>kernel[i];
    
  return;


#@cython.boundscheck(False)
#@cython.wraparound(False)
cpdef void devolve_weights(point_t[:,:] points, weight_t[:] weights, index_t[:,:] indices, sink_t[:] sink, index_t[:] shape, index_t[:] strides, int processes) nogil:
  """Converts a list of points into an volumetric image array."""
  
  cdef index_t i, j, k, v, d, n
  cdef index_t n_points  = points.shape[0];
  cdef index_t n_sink    = sink.shape[0];
  cdef index_t n_indices = indices.shape[0];                                   
  cdef index_t n_dim     = strides.shape[0];
  
  with nogil, parallel(num_threads = processes):    
    for n in prange(n_points, schedule='guided'):
      for i in range(n_indices):
        j = 0;
        v = 1;
        for d in range(n_dim):
          k = <index_t>points[n,d] + indices[i,d];
          if not (0 <= k and k < shape[d]):
            v = 0;
            break;
          else:
            j = j + k * strides[d];
        if v == 1:
          sink[j] += <sink_t>weights[n];
    
  return;


#@cython.boundscheck(False)
#@cython.wraparound(False)
cpdef void devolve_weights_intensity(point_t[:,:] points, weight_t[:] weights, index_t[:,:] indices, sink_t[:] sink, sink_t[:] weight_sum, index_t[:] shape, index_t[:] strides, int processes) nogil:
  """Converts a list of points into an volumetric image array, averaging the signal locally using weights."""

  cdef index_t i, j, k, v, d, n
  cdef index_t n_points  = points.shape[0];
  cdef index_t n_sink    = sink.shape[0];
  cdef index_t n_indices = indices.shape[0];
  cdef index_t n_dim     = strides.shape[0];

  with nogil, parallel(num_threads = processes):
    for n in prange(n_points, schedule='guided'):
      for i in range(n_indices):
        j = 0;
        v = 1;
        for d in range(n_dim):
          k = <index_t>points[n,d] + indices[i,d];
          if not (0 <= k and k < shape[d]):
            v = 0;
            break;
          else:
            j = j + k * strides[d];
        if v == 1:
          sink[j] += <sink_t>weights[n];
          weight_sum[j] += <sink_t>1;  # Increment the weight sum at the sink location

  return;
  
  
#@cython.boundscheck(False)
#@cython.wraparound(False)
cpdef void devovle_weights_kernel(point_t[:,:] points, weight_t[:] weights, index_t[:,:] indices, kernel_t[:] kernel, sink_t[:] sink, index_t[:] shape, index_t[:] strides, int processes) nogil:
  """Converts a list of points into an volumetric image array."""
  
  cdef index_t i, j, k, v, d, n
  cdef index_t n_points  = points.shape[0];
  cdef index_t n_sink    = sink.shape[0];
  cdef index_t n_indices = indices.shape[0];                                   
  cdef index_t n_dim     = strides.shape[0];
  
  with nogil, parallel(num_threads = processes):    
    for n in prange(n_points, schedule='guided'):
      for i in range(n_indices):
        j = 0;
        v = 1;
        for d in range(n_dim):
          k = <index_t>points[n,d] + indices[i,d];
          if not (0 <= k and k < shape[d]):
            v = 0;
            break;
          else:
            j = j + k * strides[d];
        if v == 1:
          sink[j] += <sink_t>(weights[n] * kernel[i]);
    
  return;
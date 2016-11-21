
import numpy as np

cimport cython

cimport numpy as np

# from libcpp.vector cimport vector

cdef dict bases={ 'A':<int>0, 'C':<int>1, 'G':<int>2, 'T':<int>3 }
cdef list bk=bases.keys()

@cython.boundscheck(False)
def add_counts( np.ndarray[long,ndim=2] counts, str string ):
    cdef int i
    for i in range(len(string)):
        if string[i] in bk:
            counts[ i, bases[ string[i] ] ] += 1L

@cython.boundscheck(False)
def seq_log_prob( np.ndarray[double,ndim=2] log_prob_mat, str string ):
    cdef int i
    assert( len(string) == log_prob_mat.shape[0] )
    res=0.0
    for i in range(len(string)):
        if string[i] in bk:
            res += log_prob_mat[ i, bases[ string[i] ] ]
    return(res)

@cython.boundscheck(False)
def one_hot( str string ):
    cdef np.ndarray[np.float32_t, ndim=1] res = np.zeros( len(string)*4, dtype=np.float32 )
    cdef int j
    for j in range(len(string)):
        res[4*j+bases[ string[j] ] ]=float(1.0)
    return(res)

@cython.boundscheck(False)
def one_hot_v( strings ):
    cdef str cat = "".join(strings)
    #cdef np.ndarray[np.float32_t, ndim=1] 
    res = one_hot( cat )
    res.shape=(len(strings), len(strings[0])*4 )
    return(res)

@cython.boundscheck(False)
def one_hot_mat( str string ):
    cdef np.ndarray[np.float32_t, ndim=2] res = np.zeros( (len(string),4,), dtype=np.float32 )
    cdef int j
    for j in range(len(string)):
        res[j,bases[ string[j] ] ]=float(1.0)
    return(res)

@cython.boundscheck(False)
def one_hot_mat_N( str string ):
    cdef np.ndarray[np.float32_t, ndim=2] res = np.zeros( (len(string),4,), dtype=np.float32 )
    cdef int j
    for j in range(len(string)):
        if string[j] in bases:
            res[j,bases[ string[j] ] ]=float(1.0)
    return(res)

import cython
import math
import sys
cimport cython
from libc.stdlib cimport malloc, free
cdef extern from "math.h":
    double sqrt(double x)
    
cdef extern from "random_sampling.c":
    void ProbSampleNoReplace(int, double *, int *, int , int *)
    
"""
def ProbSampleNoReplace(int n, double p, int nans, int ans):
"""

def sample(int n, list prob, int k):
    assert k <= n
    cdef int * perm = <int *> malloc( n * cython.sizeof(int) )
    cdef int * c_ans = <int *> malloc( k * cython.sizeof(int) )
    cdef double * prob_c = <double *> malloc( n * cython.sizeof(double) )
    cdef int i
    for i in xrange(0, n):
        prob_c[i] = prob[i]
    ans = []
    ProbSampleNoReplace(n, prob_c, perm, k, c_ans)
    
    i = 0
    for i in xrange(0, k):
        ans.append(c_ans[i])
    return ans

def list_norm(list values):
    """
    input: list of floats
    returns: vector proportional to values, sums to 1
    """
    cdef unsigned int N = len(values)
    cdef float sum_of_values = values[0]
    cdef unsigned int i
    for i in xrange(1,len(values)):
        sum_of_values += values[i]
    
    #cdef float * norm_list = <float *>malloc( len(values) * cython.sizeof(float) )
    #if norm_list is NULL:
    #    raise MemoryError()
    cdef unsigned int j    
    for j in xrange(0,N):
        values[j] = values[j] / sum_of_values
        
    return values
    
    
def is_point_in_box(float left, float right, float bottom, float top, float x, float y):
    if x >= left and x <= right:
            if y >= bottom and y <= top:
                return True
    return False
    

def jaccard_coefficient(list A, list B):
    """
    Input: two lists to be treated as sets
    """
    cdef set setA = set(A)
    cdef set setB = set(B)
    cdef unsigned int len_intersection = len(setA & setB)
    cdef unsigned int len_sum = len(setA | setB)
    cdef float result = float(len_intersection / len_sum)
    return result
 



         
def euclidean_distance(float x0, float y0, float x1, float y1):
    """
    input: x0, y0, x1, y1 - point parameters
    return  euclidean distance between points x and y
    """
    # this is 70% faster than sqrt( (x0-x1)**2 + (y0-y1)**2 )
    return math.hypot(x0-x1, y0-y1)

 #   return pow( pow(x0-x1,2) + pow(y0-y1,2) , 0.5 )

 #   return math.sqrt( (x0-x1)**2 + (y0-y1)**2 )
 
def euclidean_distance_simple(double x0, double y0, double x1, double y1):   
    cdef double xd = x0 - x1, yd= y0 - y1
    return( sqrt((xd*xd) + (yd*yd)) )
    
def mean(list vector):  
    cdef unsigned int i, N=len(vector)
    cdef double x, s=0.0
    
    for i in xrange(0,N):
       x = vector[i]
       s += x
    return s/N
        
    
    

    
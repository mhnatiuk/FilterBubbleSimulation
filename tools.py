import numpy as np
from decorators import timeit 
from math import sqrt
from math import hypot

import bottleneck as bn
from collections import defaultdict
import cython_helpers

TIMES = defaultdict(list)

#@timeit(TIMES)
def euclidean_distance(x0, y0, x1, y1):
    """
    input: x0, y0, x1, y1 - point parameters
    return  euclidean distance between points x and y
    """
    # this is 70% faster than sqrt( (x0-x1)**2 + (y0-y1)**2 )
    return hypot(x0-x1, y0-y1)
    #return cython_helpers.euclidean_distance(x0,y0,x1,y1)

def is_point_in_box(left, right,bottom,top, x, y):
    return cython_helpers.is_point_in_box(left,right,bottom,top,x,y)
    
#@timeit(TIMES)
def list_norm(values):
    """
    input: list of floats
    returns: vector proportional to values, sums to 1
    """
    sum_values = float(sum(values))
    return [ val/ sum_values for val in values ]
    
#@timeit(TIMES)    
def normalize_listnp(values):
    """
    input: list of floats
    returns: vector proportional to values, sums to 1
    """
    sum_values = float(sum(values))
    return values / sum_values
    
if __name__ == '__main__':
    for i in range(0,1000000):
        euclidean_distance(0,0.986584848,0.0001,0.999995555)
    print [("Func name:", k, "Avg time:", np.mean(v),"Total time", sum(v), "Runs:",len(v)) for k,v in TIMES.iteritems()]

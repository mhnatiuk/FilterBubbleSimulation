# -*- coding: utf-8 -*-
"""
Modification of Andreas Jung' function
https://www.andreas-jung.com/contents/a-python-decorator-for-measuring-the-execution-time-of-methods

@author Mikołaj Hnatiuk
@www www.mikolajhnatiuk.com
"""

import time                                                
from collections import defaultdict
from numpy import mean
"""
Decorator function to measure execution time of multipe function calls.
Input: defaultdict(list)
Returns: Whatever the passed function wants
Modifies Input dict!
"""
def timeit(time_dict): ## collections.defaultdict(list)
    def wrap(f):
        def wrapped_f(*args, **kw):
            ts = time.time()
            result = f(*args, **kw)
            te = time.time()
            time_dict[str(f.__name__)].append(te-ts)
            return result
        return wrapped_f
    return wrap
    

def print_bench(TIMES):
    print [("Fun:", k, "Avg time:", mean(v),"Total time", sum(v), 
            "Runs:",len(v)) for k,v in TIMES.iteritems()]
    
def unittest_timeit():

    t = defaultdict(list)
    
    @timeit(t)
    def fun(x):
        time.sleep(0.3)
        return 1
            
    i = 0
    while i < 10:    
        fun("aaa") 
        i+=1

    if abs(mean(t.values()) - 0.3) > 0.02:
        raise Exception("Mean of execution time for test function should be ~ 0.3 <+-0.02>")
    
    print "Unit test OK"        
    return t    
        
        
if __name__ == '__main__':

            
    avg_time = unittest_timeit()
    for k,v in avg_time.iteritems():
        print "Function, arguments: %s, avg exec time %f, %d runs" % (k, mean(v), len(v))
    
        
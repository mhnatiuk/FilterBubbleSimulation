from collections import defaultdict
import numpy as np
from numpy.random import normal as rnorm
from numpy.random import randint as rint
#import pylab


def vec_range(x):
    return abs(max(x) - min(x))

class preference_distributor(object):
    def __init__(self, points,sdv=1.2, number_of_points = 1000):
        """
        sdv = standard deviation of distribution
        points = list of form [ [x0,y0], [x1,y1] ... [xn,yn] ] that defines centroids of distribution
        """
        self.number_of_points = number_of_points
        self.distribution = defaultdict(list)

        for i in xrange(0,self.number_of_points):
            point = points[i % len(points)]
            self.distribution["x"].append( rnorm(loc=point[0],scale=sdv ))
            self.distribution["y"].append( rnorm(loc=point[1], scale=sdv))
        self.self_adjust()
    def self_adjust(self):
        """
        adjust distribution of points to be in range (0,1)
        modifies self.distribution
        """
        x_range = vec_range(self.distribution["x"])
        y_range = vec_range(self.distribution["y"])
        self.distribution["x"] = [ (float(el) - min(self.distribution["x"]) ) / x_range for el in self.distribution["x"]  ]
        self.distribution["y"] = [ (float(el) - min(self.distribution["y"]) ) / y_range for el in self.distribution["y"]  ]

        
    def assign(self):
        """
        Select 2 numbers [x,y] from 2D normal distribution with N centroids
        returns: list = [x,y]
        """
        point_nr = rint(low=0,high=self.number_of_points)
        point = [ self.distribution["x"][point_nr], self.distribution["y"][point_nr] ]
        # Dirty fix to avoid floating-point error causing params to exceed 1
        if point[0] <= 1.0 and point [0] >= 0.0 and point[1] >= 0.0 and point[1] <= 1.0:
            return point
        else:
            return self.assign()
    """    
    def plot_points(self):
        pylab.figure(1)
        pylab.plot(self.distribution["x"], self.distribution["y"], "o")
        pylab.show()
    """
if __name__ == "__main__":
    pd = preference_distributor(points=[[0.1, 0.1],[0.6,0.1],[0.1,0.6],[0.8,0.8]], sdv=0.1, number_of_points=1000)
    #pd.plot_points()
    print "WHAT"
    dist = dict()
    dist['x'] = []
    dist['y'] = []
    for i in range(200):
        point =pd.assign()
        dist['x'].append( point[0] )
        dist['y'].append( point[1] )  
    #pylab.figure(2)
    #pylab.plot(dist['x'],dist['y'],"o")
    #pylab.show()    
      
    
        



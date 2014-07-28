from numpy.random import normal as rnorm
from random import random as randomFloat
from random import randint as randomInt
from random import seed as randomSeed
import random
import numpy
import tools 
from collections import defaultdict

from decorators import timeit
from decorators import print_bench
from cython_helpers import *


TIMES = defaultdict(list)

class Headline:
    """
    This class stores parameters of a headline
    Headline of news is related to the news -> it is a rectangle around X and Y of news.
    However, news is not always in the center of this rectangle - it is a random process, so in fact,
    news X and Y don't even need to be inside that rectangle.
    Nx + N(0,sd) - [(1/2) * U(0.05,0.4)] # headline's left border

    """
    @timeit(TIMES)
    def __init__(self,newsX,newsY,stdev=0.5,headline_lb=0.05,headline_ub=0.4):
        """
        Input X and Y of news and standard deviation of left and right
        """
        self.newsX = newsX
        self.newsY = newsY

        self.width = random.uniform(headline_lb,headline_ub) # width of a headline box
        self.height = random.uniform(headline_lb,headline_ub) # height
        self.left = ( (self.newsX ) + random.normalvariate(mu=0,sigma=stdev) ) - 0.5*self.width # this regulates the deviation of headline from orignal news. Set it high and the estimation error goes up
        if self.left < 0:
            self.left = 0
        self.bottom = ( (self.newsY ) + random.normalvariate(mu=0,sigma=stdev) ) - 0.5*self.height
        if self.bottom < 0:
            self.bottom = 0
        self.right = self.left + self.width
        self.top = self.bottom + self.height
        
        self.diagonal = euclidean_distance_simple(self.left, self.bottom, self.right, self.top)
        

    def isNewsInHeadline(self):
        return self.isPointInHeadline(self.newsX, self.newsY)
    def isPointInHeadline(self,x,y):
        """
        Takes x and y floating-point parameters of point
        Returns: True if point is inside headline box, False otherwise
        """
		
        """
        if x >= self.left and x <= self.right:
            if y >= self.bottom and y <= self.top:
                return True
        return False
        """
		
        return is_point_in_box(self.left, self.right,self.bottom,self.top, x, y)
        
    def printValues(self):
        return "Headline Box: width: %.4f, height: %.4f \\\\ left %.4f, right %.4f, bottom %.4f, top %.4f, diag: %.4f" % (self.width,self.height, self.left, self.right, self.bottom, self.top, self.diagonal)
    def __str__(self):
        return self.printValues()
        
class News():
    def __init__(self,offer,headline_stdev=0.1, headline_lb=0.05,headline_ub=0.4):
        self.id = id(self)
        self.X = self.rnorm_range(mean=offer[0], stdev=headline_stdev , min_val=0, max_val=1 )
        self.Y = self.rnorm_range(mean=offer[1], stdev=headline_stdev , min_val=0, max_val=1 )
        

        self.headline = Headline(self.X,self.Y,headline_stdev, headline_lb,headline_ub)
    def __str__(self):
        return "News X: %s, News Y: %s, %s, News in headline:%s" % (self.X, self.Y , self.headline, self.headline.isNewsInHeadline())
        
    def rnorm_range(self, mean, stdev, min_val, max_val):
        number = rnorm(loc=mean, scale=stdev)
        if min_val <= number <= max_val:
            return number
        else:
            return self.rnorm_range(mean,stdev, min_val, max_val)
            
        
            
def testPoints(news):
    for n in news:
        assert n.X >= 0 and n.X <= 1, n.X
        assert n.Y >= 0 and n.Y <= 1, n.Y
        #print "bottom %.4f, left %.4f newsX %.4f newsY%.4f" % (n.headline.left, n.headline.bottom)
        assert (n.headline.right - n.headline.left - n.headline.width) < 0.001, (n.headline.right , n.headline.left , n.headline.width)
        assert (n.headline.top - n.headline.bottom - n.headline.height) < 0.001, (n.headline.top , n.headline.bottom , n.headline.height)
        # Test box borders:
        assert n.headline.isPointInHeadline(n.headline.left, n.headline.bottom) 
        assert n.headline.isPointInHeadline(n.headline.right, n.headline.top)
        #Test if point that must be inside a box is really there
        assert n.headline.isPointInHeadline(random.uniform(n.headline.left, n.headline.right), random.uniform(n.headline.bottom, n.headline.top) )
        # Test point that must be outside
        #assert n.headline.isPointInHeadline(random.uniform( n.headline.right, 1.0),random.uniform( n.headline.top,1.0) ) == False
    
                        
if __name__ == '__main__':
    news = [News([randomFloat(),randomFloat()], 0.1) for i in range(0,20000)]
    testPoints(news)
    
    arr = [n.headline.isNewsInHeadline() for n in news]
    #for n in news:
    #    print n
    print arr.count(True) / float(len(arr))
    print_bench(TIMES)
    
    
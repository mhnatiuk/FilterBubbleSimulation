

from collections import defaultdict
from numpy import intersect1d as intersect
from numpy import union1d as union

from numpy import nan
from numpy import mean
import pylab

from cython_helpers import mean as cython_mean
from cython_helpers import jaccard_coefficient
def get_unique_values_from_dict_of_lists(dict_of_lists):
    unique_dict = dict()
    for k,a_list in dict_of_lists.iteritems():
        for elem in a_list:
            unique_dict[elem] = 1
    return unique_dict.keys()
    

class twomode_network_data(object):
    def __init__(self, collect_network=False):
        """
        """
        self.collect_network = collect_network
        self.two_mode = defaultdict(list)
        self.params = dict()
        
    def collect(self, consumer, object_id=None):
        if consumer.id not in self.params:
            self.params[ consumer.id ] = [ consumer.X, consumer.Y ]
    def collect_twomode(self, consumer, object_id):
        #if object_id != None and object_id not in self.two_mode[ consumer.id ]:
        self.two_mode[ consumer.id ].append(object_id)
            
            
    def __str__(self):
        """
        print two mode network as an edgelist
        """
        txt = ""
        for vertex,e_list in self.two_mode.iteritems():
            for edge in e_list:
                txt += "{0} {1} ".format(vertex,edge)
        return txt

    def apply_to_edges(self, vert, data):
        edge_vals = []
        """
        def jackard_coefficient(A, B):
            inter = len(intersect(A,B))
            un = len(union(A,B))
            assert un > 0
            return inter/float(un)
        
        def jaccard_coefficient(A,B):
            A = set(A)
            B = set(B)
            inter = len(A & B)
            un = len(A | B)
            return inter/float(un)
        """    
        for i, l in enumerate(vert):
            vert_i = vert[i]
            for j in range(i, len(vert) ):
                if vert_i != vert[j]:
                    edge_vals.append( jaccard_coefficient(data[ vert_i ], data[vert[j] ] ) )
        return edge_vals
    
    def get_onemode_network_edgelist(self, ids_to_vert_nums_dict):
        self.one_mode = defaultdict(list) # vert-vert list
        self.numeric_ids = ids_to_vert_nums_dict
            
        vert = sorted(self.two_mode.keys())
        for i, l in enumerate(vert):
            vert_i = vert[i]
            vert_i_set = set(self.two_mode[vert_i])
            for j in range(i, len(vert) ):
                vert_j = vert[j]
                if vert_i != vert_j:
                    vert_j_set = set(self.two_mode[vert_j])
                    """
                    print self.two_mode
                    print vert[i]
                    print vert[j]
                    print " i: " ,self.two_mode[vert[i]], " j: ",self.two_mode[vert[j]]
                    """
                    if len( vert_i_set & vert_j_set ) > 0:
                        yield (self.numeric_ids[ vert_i ], self.numeric_ids[ vert_j ])
                        
    def save_onemode_network_edgelist(self, edgelist_fname, ids_to_vert_nums_dict):
        edgelist_fh = open(edgelist_fname, "w")
        for edge in self.get_onemode_network_edgelist(ids_to_vert_nums_dict):
             edgelist_fh.write("{0} {1} ".format(edge[0], edge[1]))
        edgelist_fh.close()
        
    def save_vertex_position(self,vertex_postiton_fname, ids_to_vert_nums_dict):
        vertex_pos_fhandle = open(vertex_postiton_fname, "w")
        x,y = zip(*self.params.values())
        #pylab.figure(1)
        #pylab.plot(x,y,"o")
        #pylab.show()
        for vertex_id, coords in self.params.iteritems():
            vertex_pos_fhandle.write("{0},{1},{2}\n".format( ids_to_vert_nums_dict[vertex_id], coords[0], coords[1]) )
        vertex_pos_fhandle.close()
    #@profile        
    def calculate_weighted_density_of_two_mode_net(self):
        """
        Calculate weighted density between one mode of a two mode network
        """
        #print "numeric id of {0} is {1} ".format('99',numeric_ids['99'])
        
        edge_values = self.apply_to_edges(  sorted(self.two_mode.keys()) , self.two_mode )
        
        if len(edge_values) > 1 :
            mean_of_edge_values = cython_mean(edge_values)	
        else:
            mean_of_edge_values = nan
        
        #print "Users:" , self.two_mode.keys()
        #print "News:" , get_unique_values_from_dict_of_lists( self.two_mode )  
        #max_possible_sum = len(get_unique_values_from_dict_of_lists( self.two_mode )) * len(self.two_mode.keys())
        return mean_of_edge_values
        
    def calc_wdens_2mode(self):
        return self.calculate_weighted_density_of_two_mode_net()
        
if __name__ == '__main__':
    import uuid
    from random import random as randomFloat
    class Consumer(object):
        def __init__(self,params):
            self.id = uuid.uuid4()
            self.X = params[0]
            self.Y = params[1]
            
    def test_normal_case():
        netdata = twomode_network_data()
        c1,c2,c3,c4,c5 = [ Consumer([randomFloat(),randomFloat()] ) for i in range(0,5) ]
        
        netdata.collect(c1, "A")
        netdata.collect(c1, "B")
        netdata.collect(c2, "A")
        netdata.collect(c2, "B")
        netdata.collect(c3, "B")
        netdata.collect(c4, "C")
        netdata.collect(c5, "D")
        #print netdata
        #print netdata.calculate_weighted_density_of_two_mode_net() # transform after collection has finished
        return True
    def test_0_nodes():
        netdata = twomode_network_data()
        try:
            netdata.calculate_weighted_density_of_two_mode_net()
        except ZeroDivisionError:
            return ZeroDivisionError
        return True
    
    assert test_normal_case() == True, "Test failed!"
    assert test_0_nodes() == True, "Division by zero. Test failed!"
    for i in xrange(1000):
        test_normal_case()
    
    
    #print netdata.calculate_density()
    
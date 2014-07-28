
import itertools
from random import random
from collections import defaultdict
from collections import deque
import numpy as np
import bottleneck as bn
from pandas import DataFrame
from pandas import Series

   

def makeup_missing_data(data):
    ticks = sorted(data.keys())
    for tick in xrange(min(ticks),max(ticks)):
        if tick not in ticks:
            data[tick] = [np.nan]

class DataCollector(object):
    """
    DataCollector collects data from simulation.
    """
    def __init__(self,fun_dict):
        self.data = defaultdict(lambda: defaultdict(list))
        self.data_per_agent = defaultdict(lambda: defaultdict( lambda: defaultdict(list) ) )
        self.functions = {}
        self.aggregate_functions(fun_dict)
        
    def aggregate_functions(self, fun_dict):
        for var_name, fun in fun_dict.iteritems():
            self.functions[var_name] = fun
    #@profile
    def collect(self, var_name, value, tick):
        """
        Collects value for specified variable and simulation tick
        """
        
        self.data[var_name][tick].append(value)
    
        #if var_name not in self.functions.keys():
        #    self.functions[var_name] = fun
    
    def collect_per_agent(self, var_name, value, tick, agent_id):
        #print "data: ",self.data_per_agent[var_name][agent_id]
        #print "tried: ", var_name, value, tick, agent_id
        if type(value) == float or type(value) == int:
            self.data_per_agent[var_name][agent_id][tick].append(value)
        if type(value) == list:
            self.data_per_agent[var_name][agent_id][tick].extend(value)

        
    def make_dataframe(self, max_ticks):
        """
        Returns pandas data frame. Rows are ticks, columns - variables
        """
        ticks_data = self.data[self.data.keys()[0]].keys()
        #makeup_missing_ticks(ticks_data)
        ticks_column = { 'tick' : Series( xrange(min(ticks_data), max(ticks_data)+1 ) ) }
        
        columns = {}
        
        def make_standard_data(columns):
            for var_name, values in self.data.iteritems():
                fun = self.functions[var_name]
                #makeup_missing_data(self.data[var_name])
                columns[ var_name ] = Series([fun(v) for v in self.data[var_name].values()])
                
        def make_agent_columns(columns):                
            for var_name, agents in self.data_per_agent.iteritems():

                # for each tick aggregate (np.mean) variances for each agent
                # data returned from fun (if fun is moving_variance) are variances for each tick in form of a list
                # [0.2, 0.5 ... x]. Access each tick
                window_size = 5
                variances = {}
                for agent_id, ticks in agents.iteritems():
                    #print "Before ",
                    #print agents[agent_id].keys()
                    #makeup_missing_data(agents[agent_id])
                    #print "After ",
                    #print agents[agent_id].keys()
                    for tick in sorted(agents[agent_id].keys()):
                        agent = agents[agent_id]
                        if tick >= window_size :
                            start = tick - (window_size - 1)
                            #print "calc ", start
                        else: 
                            start = 0
                        end = tick + 1
                        #print tick, " -> ", start,":",end
                        collector = []
                        #print "DATA: ",agents[agent_id]
                        if start != end:
                            for i in xrange(start,end):
                                collector.extend(agent[i])
                        else:
                            collector.extend(agent[tick])
                            
 #                       collector.extend(agents[agent_id][tick][start:end] )
                        #print collector
                        try:
                            #print "Start {0}  end {1} tick {2}".format(start, end, tick)
                            #print "len: ", len(tick_data), " data: ", tick_data
                            #print "full : ",agents[agent_id][tick]
                            #print "Var: ",bn.nanvar(tick_data),

                            variances[tick].append(bn.nanvar(collector))
                        except KeyError:
                            variances[tick] = []
                            #print "key error, collector:", collector
                            #print collector
                            variances[tick].append(bn.nanvar(collector))
                        

                columns[ var_name ] = Series([bn.nanmean(variances_list) for variances_list in variances.values()])
                
                
        make_standard_data(columns)        
        make_agent_columns(columns)
        #print "ticks,",ticks_column.items()
        #print "cols,",columns.items()

        return DataFrame(dict(ticks_column.items()+ columns.items()))
        

if __name__ == '__main__':
    
    dc= DataCollector({'var1':bn.nanmean, 'var2':bn.nansum, 'var3':bn.nanmean,
                        'var4': None})
    ticks = 10
    """ testdata = {0 : {0:[0,0], 1:[0,0]}, 1 : {0:[1,1], 1:[2,2]}, 2 : {0:[1,1], 1:[2,2]}, 3: {0:[1,1], 1:[2,2]}, 4 : {0:[1,1], 1:[2,2]}, 5 : {0:[1,1], 1:[2,2]},
                6 : {0:[1,1], 1:[2,2]}, 7 : {0:[1,5], 1:[2,10]}, 8 : {0:[2,15], 1:[0,20]}, 9 : {0:[1,100], 1:[2,200]}, 10:{0:[1,100], 1:[2,2200]}}
                """
    testdata = {0: {0:[1,1], 1:[1,1], 2:[1,1], 3:[1,1], 4:[1,1], 5:[0,20], 6:[0,20], 7:[0,20], 8:[0,20], 9:[0,20] },
                1: {0:[1,1], 1:[1,1], 2:[1,1], 3:[1,1], 4:[1,1], 5:[0,20], 6:[0,20], 7:[0,20], 8:[0,20], 9:[0,20] }
                }
    
    counter = 300
    while counter > 0:
        counter -= 1
        for tick in xrange(ticks):
            if tick == 5:
                continue
    
            dc.collect('var1', [1.0,2.0], tick)
            dc.collect('var2', 3, tick)
            dc.collect('var3', [10.0,10.0], tick)
            for agent in xrange(2):
                dc.collect_per_agent('var4', testdata[agent][tick][0], tick, agent)
                dc.collect_per_agent('var4', testdata[agent][tick][1], tick, agent)
        df = dc.make_dataframe(ticks)
        #print "Testing moving variance feature ..."
        #assert round(df["var4"][9], 2 ) == 100.0
        #print df
        #print "TEST OK!"
            
        
        
        
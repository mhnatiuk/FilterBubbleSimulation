from __future__ import division
#from __future__ import print_function
import itertools
from random import random as randomFloat
from random import randint as randomInt
from random import seed as randomSeed
import random
import time
import csv
from collections import defaultdict
from collections import deque
import numpy
from numpy import nan
import pandas as pd
from pandas import Series
import uuid

import ipdb

# Custom modules
from preference_distributor import preference_distributor
from twomode_network_data import twomode_network_data
from news import *

from data_collector import *
from decorators import timeit
from cython_helpers import mean as cython_mean
from cython_helpers import euclidean_distance_simple, list_norm, sample

TIMES = defaultdict(list)
wd =''




"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

                            Model of Consumer

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

class Consumer:
    def __init__(self,wtr, preferences, producers, data_collector, delete_cookies_ratio, producers_to_choose):
        self.id = id(self)
        self.X = preferences[0]
        self.Y = preferences[1]
        self.willToRead = float(wtr)
        
        self.producers = producers
        self.producers_ranking = defaultdict(int)
        
        self.consumed_news_history = 5
        self.consumed_news_params = {"X": dict(), "Y": dict() }     
            
        self.producers_ranking = dict( (pr.id,1) for pr in self.producers)
        self.data_collector = data_collector
        self.delete_cookies_ratio = delete_cookies_ratio
        self.producers_to_choose = producers_to_choose

    
    def step(self,tick, network_data):
        self.tick = tick
        self.network_data = network_data
        consumed = 0
        too_far = 0
        consumed_news_id = list()        

    
        wtr= self.willToRead * ( randomFloat() > 0.6 ) # probability that 
        #        WTR == 0 is 0.6 so only 40% of consumers consume at each turn
        if wtr == 0:
            return None

    
        if self.delete_cookies_ratio != None and (self.tick < 120 or self.tick > 500): # ~!~~~~~~~~~~~~~~~~~~~~
            delete_cookies = randomFloat() < self.delete_cookies_ratio # Delete cookies?
        else:
            delete_cookies = False
            
        
        """
        Branch: headlines: consumer doesn't know exact parameters of news 
            until it's read. He/she can scan headlines which give some idea
            about what are the parameters of news. If user parameters are 
            inside "the headline box", user decides to read it
        """
        def consumption_probability(ranking):
         # vector of probabilities of consumption from that producer
            return dict( (k,v) for k,v in zip(ranking.keys(),
                    list_norm(ranking.values()) ) )


        prob_consumption = list_norm(self.producers_ranking.values()) #consumption_probability(self.producers_ranking)
        #if type(self.producers_to_choose) == float:
        choices = sample(len(self.producers_ranking), prob_consumption, self.producers_to_choose)         
        #if inspect.isfunction(function):
        #    choices = sample(len(self.producers_ranking), prob_consumption, self.producers_to_choose() )         
        producers_chosen = [producer for i, producer in enumerate(self.producers) if i+1 in choices] # i+1 because cython_helpers.sample returns ints from 1, not 0

        
        news_offered = 0
        number_of_producers_chosen = 0
                
        #print wtr
        for pr in producers_chosen: #self.producers:
            if delete_cookies == True:
                pr.delete_history(self.id)
#            if prob_consumption[pr.id] > randomFloat() and wtr > 0:                
            infoset = pr.get_info(self.id) # get recommended news
            news_offered += len(infoset)                    
            number_of_producers_chosen += 1                    

            for news in infoset:
                                            
                # calculate real distance between news and preferences
                real_distance = euclidean_distance_simple(self.X, self.Y, news.X, news.Y )                             
                self.data_collector.collect('avg_distance_all', real_distance,
                        self.tick)                                    
                # read news if headline suggests that this article is
                # interesting to the user:                        
                if news.headline.isPointInHeadline(self.X, self.Y): 
                    consumed += 1    
                    self.consume(news, pr)    
                    self.data_collector.collect_per_agent('var_of_consumed_x', news.X, self.tick, self.id)
                    self.data_collector.collect_per_agent('var_of_consumed_y', news.Y, self.tick, self.id)
                    #self.data_collector.collect('est_err_X',est_err_X ,self.tick)    
                    consumed_news_id.append(news.id)                            
                    self.data_collector.collect( 'avg_distance_consumed', 
                            real_distance, self.tick )    
                    if real_distance < news.headline.diagonal :
                        wtr -= 0
                        self.award_producer(pr.id)
                    else:
                        wtr -= 2
                        too_far += 1
                        break
                else:
                    wtr -=1           
            #block: after consuming information. collect estimation error made by producer
            self.save_estimation_error(pr)
            
        self.data_collector.collect('avg_number_of_producers_chosen', number_of_producers_chosen, self.tick)
        try:
            self.data_collector.collect('avg_read_from_offer', 
                consumed / float(news_offered) , self.tick)
        except ZeroDivisionError:
            self.data_collector.collect('avg_read_from_offer', nan, self.tick)
                    
        self.data_collector.collect('consumed_news', consumed, self.tick)
        self.data_collector.collect('too_far', too_far, self.tick)
        self.data_collector.collect('wtr_at_end', wtr, self.tick)        
#@timeit(TIMES)

    def save_estimation_error(self,pr):
        est_X = pr.get_estX(self.id)
        est_Y = pr.get_estY(self.id)        

        if type(est_X) == float:
            est_err_X = abs(est_X - self.X)
        else:
            est_err_X = nan     

        if type(est_Y) == float:
            est_err_Y = abs(est_Y - self.Y)
        else:                   
            est_err_Y = nan
        self.data_collector.collect('est_err_X', est_err_X ,self.tick)
        self.data_collector.collect('est_err_Y', est_err_Y, self.tick) 
    def award_producer(self,pr_id):
        """
        Increment points in consumer's producers ranking
        """
        self.producers_ranking[pr_id] += 1 
        
    def consume(self,news,pr):
        pr.ack(self.id, news)
        self.network_data.collect_twomode(self, news.id)
  
        # increment the number of consumed news from that producer:
    #@timeit(TIMES)
    def prob_function(self,x):
        xnorm = x * 12 - 6
        return (0.01) / ( 0.01 + ( numpy.e ** xnorm )**2 )
        

    def __str__(self):
        return "Id: {0} X:{1} Y:{2}".format(self.id,self.X,self.Y, self.willToRead)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

                            Model of Producer

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

class Producer:
    def __init__(self, newsNo,headline_stdev,headline_lb,headline_ub, params, profiling_start, start_data_collection, data_collector) :
        self.id = id(self)
        self.newsNo = newsNo
        self.news = []
        self.users = defaultdict(dict)
        self.params = params
        self.headline_stdev = headline_stdev
        self.headline_lb = headline_lb
        self.headline_ub = headline_ub
        self.profiling_start = profiling_start
        self.start_data_collection = start_data_collection
        self.data_collector = data_collector
        self.number_of_consumed_news = 0
    def step(self,tick):
        self.tick = tick
        self.data_collector.collect('avg_number_of_consumed_news', self.number_of_consumed_news, self.tick)
        self.number_of_consumed_news = 0
        self.news = [News(self.params, self.headline_stdev, self.headline_lb,self.headline_ub) for i in xrange(self.newsNo) ]
    
    #@timeit(TIMES) 
    def get_info(self, uid):

        if uid not in self.users:
            self.new_user(uid)
        
        if len(self.users[uid]["X"]) > 0 and self.tick > self.profiling_start :               
            dist = self.calculate_distance(uid) #timed
            return self.sort_news(self.news, dist)[:4] #timed
        else:
            return self.news[:4]
        
        return self.news[:4]
    #@timeit(TIMES)
    def sort_news(self, news,dist):
        # this happens to be called a Schwartzian transformation        
        return [ news for (dist, news) in sorted(zip(dist, news)) ]  

               
    #@timeit(TIMES)
    def calculate_distance(self, uid):
        """
        Calculates distance of news parameters to current consumer's preference estimation
        Input: Consumer's id
        Returns: list of euclidean distances for each news in self.news
        """
        #assert 0 < self.users[uid]["estX"] < 1, "estimation exceeds 0 and 1"
        return [ euclidean_distance_simple(self.users[uid]["estX"], self.users[uid]["estY"],
                news.X, news.Y) for news in self.news ] 
        
    def new_user(self, uid):
        self.users[uid]["X"] = [] # deque()
        self.users[uid]["Y"] = [] #deque()
        self.users[uid]["estX"] = nan
        self.users[uid]["estY"] = nan      
        
    def delete_history(self, uid):
        self.new_user(uid)
    #@timeit(TIMES)
    def ack(self,uid, news):
        self.number_of_consumed_news += 1
        
        #if self.tick < self.start_data_collection:
        #    return None
			
        self.users[uid]["X"].append(news.X)
        self.users[uid]["Y"].append(news.Y)
        
        self.users[uid]["X"] = self.users[uid]["X"][-10:]
        self.users[uid]["Y"] = self.users[uid]["Y"][-10:]
        
        self.data_collector.collect_per_agent('var_x_of_last_10_news', self.users[uid]["X"], self.tick, uid)
        self.data_collector.collect_per_agent('var_y_of_last_10_news', self.users[uid]["Y"], self.tick, uid)
        
        self.users[uid]["estX"] = self.__get_estX(uid)
        self.users[uid]["estY"] = self.__get_estY(uid)
        
        assert 0 <= self.users[uid]["estX"] <= 1, "estX: {0} is not between <0,1>".format(self.users[uid]["estX"])
        assert 0 <= self.users[uid]["estY"] <= 1, "estY: {0} is not between <0,1>".format(self.users[uid]["estY"])	
    def __est(self,uid,key):
        """
        Private method. Estimates consumer's preferences based on reading data
        """        
        try:
            return cython_mean(self.users[uid][key]) # 
        except ZeroDivisionError:
            return numpy.nan

    def __get_estX(self, uid):
        return self.__est(uid,"X")
        
    def __get_estY(self, uid):
        return self.__est(uid,"Y")
        
    def __get_est(self, uid, key):
        if self.users[uid].has_key(key):
            return self.users[uid][key]
        else:
            return nan
                    
    def get_estX(self, uid):
        return self.__get_est(uid, "estX")
        
    def get_estY(self, uid):
        return self.__get_est(uid, "estY")
    
    def __str__(self):
        return self.id
"""
==========================================
Simulation class
==========================================
"""        
class Sim():
    #numP=1, numC=1, ticks=1000,newsNo=40,wtr=20, pickyTick=40,
    # profiling_start=80,headline_stdev=0.1, start_data_collection=0):
    def __init__(self, **kwargs ):
        randomSeed()
        self.tick = 0
        self.producers = []
        self.consumers = []
        self.consumer_ids_to_numbers = dict()
        self.param = dict()
        for param,val in kwargs.iteritems():
            setattr(Sim, param, val)
        self.data_collector = DataCollector({'est_err_X' : bn.nanmean,
                'est_err_Y': bn.nanmean , 'too_far' : sum, 
                'consumed_news': bn.nanmean, 'avg_read_from_offer': bn.nanmean,
                'avg_distance_consumed': bn.nanmean, 'var_of_consumed_x': None,
                'var_of_consumed_y': None, 'avg_distance_all' : bn.nanmean, 
                'weighted_net_density':bn.nanmean,
                'var_x_of_last_10_news':None, 'var_y_of_last_10_news': None,
                'avg_number_of_producers_chosen': bn.nanmean,
                'wtr_at_end' : bn.nanmean , 'avg_number_of_consumed_news': bn.nanmean})

         
    #@timeit(TIMES)
    def run(self):
        self.setup()
        while self.tick < self.maxTick:
            self.step()
            #print "--------\m End of Tick %d" % self.tick
            self.tick+=1    
        ## finish
        if self.collect_network == True:
            self.network_data.save_vertex_position(wd + "sna\\vertex_position_take2.csv", self.consumer_ids_to_numbers) ## 
    
            
    
    def setup(self):
        assert self.producers_to_choose <= self.numP, "producers_to_choose MUST be <= numP"
        self.dist_consumers_pref = preference_distributor(points = self.consumer_points,sdv = self.consumer_stdev)
        self.dist_producers_offer = preference_distributor(points = self.producer_points, sdv = self.producer_stdev)  # What about distance between points of consumers and producers. Avg distance relation to consumption
        #dist_producers_pref = preference_distributor(points=[[0.1, 0.1],[0.6,0.1],[0.1,0.6],[0.8,0.8]])
        last_consumer_number = 0
        
        for i in xrange(self.numP):
            producer_xy = self.dist_producers_offer.assign()

            self.producers.append(Producer(self.newsNo,self.headline_stdev, self.headline_lb,self.headline_ub , producer_xy, self.profiling_start, self.start_data_collection, self.data_collector))
            
        for i in xrange(self.numC):
            consumer_xy = self.dist_consumers_pref.assign()
            consumer_obj = Consumer(self.wtr, consumer_xy, self.producers, self.data_collector, self.delete_cookies_ratio, self.producers_to_choose)
            last_consumer_number += 1
            self.consumer_ids_to_numbers[ consumer_obj.id ] = last_consumer_number
            
            self.consumers.append(consumer_obj)        
            
    def step(self):

        def move_producers():
            for pr in self.producers:
                pr.step(self.tick)
        def move_consumers():
            self.network_data = twomode_network_data() # data collection class
            
            for consumer in self.consumers:
                self.network_data.collect(consumer)
                consumer.step(self.tick, self.network_data)
                                                
            net_density = self.network_data.calc_wdens_2mode() 
            self.data_collector.collect('weighted_net_density', net_density, self.tick)
            if self.collect_network == True:
                self.network_data.save_onemode_network_edgelist(wd + "sna\\network."+str(self.tick)+".txt", self.consumer_ids_to_numbers)
                
        move_producers()
        move_consumers()
    def getProducer(self):
        return random.choice(self.producers)
              
    def __str__(self):
        return "Parameters" + ";".join([ k+":"+str(v) for k,v in self.params.iteritems() ])
        

def save_df(runs, wd, file_name, test_prefix='', ext='csv'):
    outfile = open(wd + test_prefix +file_name + "_aggregated"  + ext, "w")
    df_full = pd.concat(runs)
    df_mean = df_full.groupby(df_full.index).mean()
    df_mean.to_csv(outfile)
    outfile.close()
def save_params(wd, file_name, test_prefix='', ext='csv'):
    params_file = open(wd + test_prefix +"params_"+file_name + ext, "w")
    params_file.writelines(";".join([str(p) for p in args.simulation.keys() ]) + "\n")
    params_file.writelines(";".join([str(p) for p in args.simulation.values() ])+"\n")
    params_file.close()        
        
"""##########################################################################        
        main
##############################################################################"""

if __name__ == '__main__':
    import argparse
    import importlib
    
    parser = argparse.ArgumentParser(description='Run filter bubble simulation.')
    parser.add_argument('outfile', type=str, help='prefix name for the data files')
    parser.add_argument('params', type=str, help='file with parameters for simulation')
    parser.add_argument('trials', type=int, help='how many simulations to run')
    cmd_args = parser.parse_args()

    #import arguments from configuration file
    args = importlib.import_module("%s" % cmd_args.params)
    wd = args.io['wd'] # global 
    runs = []

    print ( "Run: ")
    for i in xrange(cmd_args.trials):
        print(i)
        if i > 0:
            args.simulation['collect_network'] = False
        sim =  Sim(**args.simulation)
        sim.run()
        
        df = sim.data_collector.make_dataframe(args.simulation['maxTick'])

        df.to_csv(wd + args.io['test_prefix'] + cmd_args.outfile + "_run" + str(i) + args.io['ext'])
        runs.append(df)
    


    if args.io['save_data']:
        save_df(runs, wd, cmd_args.outfile, args.io['test_prefix'], args.io['ext'])
        save_params(wd, cmd_args.outfile, args.io['test_prefix'], args.io['ext'])


    print( "Code benchmarks:" )
    print([("Fun:", k, "Avg time:", numpy.mean(v),"Total time", sum(v), 
            "Runs:",len(v)) for k,v in TIMES.iteritems()] )

        
    




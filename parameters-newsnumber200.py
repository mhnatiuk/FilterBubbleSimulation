# SimulationParameters

simulation = dict(
numP =  10,
numC = 200,
maxTick = 1000,
newsNo = 200,
wtr = 20.0,
producers_to_choose = 5, #### numP
pickyTick = 40,
profiling_start = 200,
headline_stdev = 0.6,
headline_lb=0.2,
headline_ub=0.6,
start_data_collection = 0,
collect_network = False,
delete_cookies_ratio = None,
consumer_points = [[0.1, 0.1],[0.6,0.1],[0.1,0.6],[0.8,0.8]],
producer_points = [ [0.15,0.2], [0.4, 0.2], [0,3,0.5], [0.7, 0.7] ],
consumer_stdev = 0.05,
producer_stdev = 0.05,
)



# IOParameters
io = dict(save_data = True,
wd = 'C:/Users/mikol_000/Documents/Doktorat/FilterBubble/data/',
test_prefix = '',
ext = 'csv',)

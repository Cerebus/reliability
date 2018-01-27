import time, argparse
import networkx as nx
import numpy.random as random
from scoop import futures
# from multiprocessing import Pool

# Agrest-Coull binomial proportion confidence interval estimate
# Input: Successful observations, total observations, and quantile of the standard normal
#        distribution for the desired error rate (i.e., the probit)
# Output: The lower and upper bounds of the interval
def CI(C,N,z):
    nBar = N + z**2
    cBar = (1/nBar)*(C+(z**2/2))
    l = cBar - z * ((1/nBar)*cBar*(1-cBar))**0.5
    u = cBar + z * ((1/nBar)*cBar*(1-cBar))**0.5
    return (l,u)

# Monte Carlo Reliability algorithm 3 worker function
# Input: A labeled graph, iterations to run, and a source and target node
# Output: Number of successful observations and total number of observations
#def R3worker(G,I,s,d):
def R3worker(x):
    G,I,s,d = x
    random.seed() # Gotcha:  Pool processes get the same PRNG state. Must reseed.
    C = 0
    for i in range(I):
        g = G.copy()
        to_remove = []
        for e in list(g.edges):
            if random.random() > g.edges[e]['R']:
                to_remove.append(e)
        g.remove_edges_from(to_remove)
        if nx.has_path(g,s,d):
            C += 1
    return (C,I)

# Monte Carlo Reliability algorithm 3 - parallel
# Input: A labeled graph, the confidence interval target, and a source and target node
# Output: Reliability probability, confidence interval
#def R3(G,I,s,d):
def R3(G, P, c, I, s, d):
    C = 0
    N = 1
    z = 1.96
    l,u = CI(C,N,z)
    # with Pool(processes=2) as p: 
    #print('0: ',l,u)
    while u-l > I:
        #print('loop: ',l,u)
            # results = p.starmap(R3worker, [(G,10000,s,d),(G,10000,s,d)])
        results = list(futures.map(R3worker, ((G,c,s,d),)*P))
        #print('r: ', results)
        C += sum(c for c,_ in results)
        N += sum(n for _,n in results)
        l,u = CI(C,N,z)
    return C/N, CI(C,N,z), N    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(usage='python3.6 -m scoop [scoop options] -- %(prog)s [options]',
                                     epilog='Estimate reliability over a labeled network.\n\
                                     Queue depth should be at least equal to the number of\
                                     scoop workers.',formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-p','--procs', type=int, default=1, help='queue depth')
    parser.add_argument('-c','--count', type=int, default=100, help='iterations per queue entry')
    parser.add_argument('-ci','--confint',type=float, default=0.005, help='confidence interval threshold')
    parser.add_argument('startnode', type=str, default='1', help='starting node')
    parser.add_argument('endnode', type=str, default='2', help='ending node')
    parser.add_argument('graph',type=argparse.FileType('r'), help='graph file')
    args = parser.parse_args()
    
    G = nx.read_graphml(args.graph)
    starttime = time.time()
    R, CI, N = R3(G, args.procs, args.count, args.confint, args.startnode, args.endnode)
    endtime = time.time()
    print('Trials: {}\nR: {:.5}\nLower bound: {:.5}\nUpper bound: {:.5}'.format(N, R, CI[0], CI[1]))
    print('Time: ', endtime-starttime)







    #G1 = nx.MultiGraph()
    #G1.add_edges_from([(1,2),(1,2)])
    #nx.set_edge_attributes(G1,1.0,'R')
    #nx.set_edge_attributes(G1,{(1,2,0):{'R':0.91}, (1,2,1):{'R':0.81}})

    #G2 = nx.MultiGraph()
    #G2.add_nodes_from([1,2,3,4,5,6,7,8])
    #G2.add_edges_from([(1,2),(2,7),(2,3),(3,4),(3,4),(4,5),
    #                   (5,6),(5,6),(6,7),(7,8)])
    #nx.set_edge_attributes(G2,1.0,'R')
    #nx.set_edge_attributes(G2,{(2,7,0):{"R":0.8}, (2,3,0):{"R":0.8},
    #                           (3,4,0):{"R":0.8}, (3,4,1):{"R":0.8}, 
    #                           (5,6,0):{"R":0.8}, (5,6,1):{"R":0.8}})


    
#results = []
#for i in range(10000,200000,10000):
#    start = time.time()
#    r,ci = R(G,i,1,2)
#    end = time.time()
#    results.append([i,r]+list(ci)+[ci[1]-ci[0],end-start])
#g1_results = pd.DataFrame(results,columns=['Trials','R estimate','Lower bound','Upper bound',
#                                                  'Interval','Execution time'])
#g1_results['Graph'] = 1;

#results = []
#for i in range(10000,200000,10000):
#    start = time.time()
#    r,ci = R(G2,i,1,8)
#    end = time.time()
#    results.append([i,r]+list(ci)+[ci[1]-ci[0],end-start])
#g2_results = pd.DataFrame(results,columns=['Trials','R estimate','Lower bound','Upper bound',
#                                                  'Interval','Execution time'])
#g2_results['Graph'] = 2;

#results = []
#for i in range(500):
#    start = time.time()
#    r,ci = R(G,10000,1,2)
#    end = time.time()
#    results.append([i,10000,r]+list(ci)+[ci[1]-ci[0],end-start])
#g1_error_results = pd.DataFrame(results,columns=['Run','Trials','R estimate','Lower bound','Upper bound',
#                                                  'Interval','Execution time'])
#g1_error_results['Graph'] = 1;

#total_errors = ([i for (i,r) in g1_error_results.iterrows() if 
# (r['Upper bound']<0.9829) or (r['Lower bound']>0.9829)])
#print('Total errors: {}\nError rate: {}'.format(len(total_errors), len(total_errors)/500))

#results = []
#for i in range(500):
#    start = time.time()
#    r,ci,n = R2(G,0.003,1,2)
#    end = time.time()
#    results.append([i,n,r]+list(ci)+[ci[1]-ci[0],end-start])
#g1_r2_results = pd.DataFrame(results,columns=['Run','Trials','R estimate',
#                                              'Lower bound','Upper bound',
#                                              'Interval','Execution time'])
#g1_r2_results['Graph'] = 1;

#total_errors = ([i for (i,r) in g1_r2_results.iterrows() if 
# (r['Upper bound']<0.9829) or (r['Lower bound']>0.9829)])
#print('Total errors: {}\nError rate: {}'.format(len(total_errors), len(total_errors)/500))

#results = []
#for i in range(500):
#    start = time.time()
#    r,ci,n = R3(G,0.003,1,2)
#    end = time.time()
#    results.append([i,n,r]+list(ci)+[ci[1]-ci[0],end-start])
#g1_r3_results = pd.DataFrame(results,columns=['Run','Trials','R estimate',
#                                              'Lower bound','Upper bound',
#                                              'Interval','Execution time'])
#g1_r3_results['Graph'] = 1;

#r2 = g1_r2_results['Execution time'].sum()
#r3 = g1_r3_results['Execution time'].sum()

#print("R2 total execution time: {:.2f} seconds".format(r2))
#print("R3 total execution time: {:.2f} seconds".format(r3))
#print("Speedup: {:.2f}".format(r2/r3))

#total_errors = ([i for (i,r) in g1_r3_results.iterrows() if 
# (r['Upper bound']<0.9829) or (r['Lower bound']>0.9829)])
#print('Total errors: {}\nError rate: {}'.format(len(total_errors), len(total_errors)/500))

# Save critical variables for future sessions
#import pickle
#with open('MCRE-variables-save.p','wb') as f:
#    pickle.dump([G,
#                 G2,
#                 g1_results,
#                 g2_results,
#                 g1_error_results,
#                 g1_r2_results,
#                 g1_r3_results],
#                f)

#import pickle
#with open('MCRE-variables-save.p','rb') as f:
#     G,G2,g1_results,g2_results,g1_error_results,g1_r2_results,g1_r3_results = pickle.load(f)

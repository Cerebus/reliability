import time, argparse
import networkx as nx
# Numpy random is better quality, but takes forever to build on arm32v7
#import numpy.random as random
import random
from scoop import futures
import pickle

import cProfile

def profileit(func):
    def wrapper(*args, **kwargs):
        datafn = func.__name__ + ".profile" # Name the data file sensibly
        prof = cProfile.Profile()
        retval = prof.runcall(func, *args, **kwargs)
        prof.dump_stats(datafn)
        return retval
    return wrapper


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
def R3worker(x):
    G,I,s,d = x
    random.seed() # Gotcha:  Pool processes get the same PRNG state. Must reseed.
    C = 0
    for i in range(I):
        # g = G.copy()
        # networkx does deepcopy, which could be slow.  
        g = pickle.loads(pickle.dumps(G))
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
@profileit
def R3(G, P, c, I, s, d):
    C = 0
    N = 1
    z = 1.96
    l,u = CI(C,N,z)
    while u-l > I:
        results = list(futures.map(R3worker, ((G,c,s,d),)*P))
        C += sum(c for c,_ in results)
        N += sum(n for _,n in results)
        l,u = CI(C,N,z)
    return C/N, CI(C,N,z), N    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()#usage='python3.6 -m scoop [scoop options] -- %(prog)s [options]',
                                     #epilog='Estimate reliability over a labeled network.\n\
                                     #Queue depth should be at least equal to the number of\
                                     #scoop workers.',formatter_class=argparse.ArgumentDefaultsHelpFormatter
    #)
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

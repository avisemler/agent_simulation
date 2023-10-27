"""Used to find graphs with the same number of edges 
using different generators."""

import networkx as nx

g = nx.erdos_renyi_graph(3000, 0.0053351117)
print(len(g.edges))

g = nx.watts_strogatz_graph(3000, 17, 0.1)
print(len(g.edges))

g = nx.barabasi_albert_graph(3000, 8)
print(len(g.edges))

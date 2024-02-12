"""Used to find graphs with the same number of edges 
using different generators."""

import networkx as nx


print("High")
g = nx.erdos_renyi_graph(3000, 0.0053351117)
print(len(g.edges))

g = nx.watts_strogatz_graph(3000, 17, 0.1)
print(len(g.edges))

g = nx.barabasi_albert_graph(3000, 8)
print(len(g.edges))


print("Low")


g = nx.erdos_renyi_graph(3000, 0.00133377792)
print(len(g.edges))

g = nx.watts_strogatz_graph(3000, 5, 0.1)
print(len(g.edges))

g = nx.barabasi_albert_graph(3000, 2)
print(len(g.edges))

import networkx as nx
import matplotlib.pyplot as plt
import gcol

f = open("graph_plain.txt", "r")

data = []
for line in f:
    for edge in line.split(", "):
        s = edge.split(":")
        if len(s) == 2:
            data.append({s[0], s[1]})
        
G = nx.Graph(data)

for strategy in ["random", "welsh_powell", "dsatur", "rlf"]:
    c = gcol.node_coloring(G, strategy)
    print(strategy, " ", max(c.values()) + 1)
    


c = gcol.node_coloring(G, opt_alg=2, it_limit=len(G)*100)
print("opt2 ", max(c.values()) + 1)

c = gcol.node_coloring(G, opt_alg=3, it_limit=len(G)*100)
print("opt3 ", max(c.values()) + 1)

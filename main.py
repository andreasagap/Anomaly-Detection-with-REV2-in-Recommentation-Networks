import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import scipy
def draw_graph(G):
    plt.figure()
    pos = nx.spring_layout(G)
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    nodeShapes = set((aShape[1]["s"] for aShape in G.nodes(data=True)))
    colors = ['green','blue']
    for (index,aShape) in enumerate(nodeShapes):
        nx.draw_networkx_nodes(G, pos, node_color=colors[index] ,node_shape=aShape, nodelist=[sNode[0] for sNode in
                                                                              filter(lambda x: x[1]["s"] == aShape,
                                                                                     G.nodes(data=True))])

    nx.draw_networkx_edges(G,pos,edgelist=edges,label =weights)

    #nx.draw_networkx(G, pos,with_labels=False, edges=edges,width=weights)
    plt.show()

def createGraph():
    df = pd.read_csv('ratings_musical_Amazon.csv',names=["user", "product","rating"])
    df = df.head(50)
    G = nx.Graph()
    users = df["user"].unique()
    products = df["product"].unique()
    for u in users:
        G.add_node(u, s="o")
    for p in products:
        G.add_node(p, s="^")

    for (index,row) in df.iterrows():
        G.add_edge(row[0],row[1],weight = row[2])
    draw_graph(G)
if __name__ == '__main__':
    createGraph()


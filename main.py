import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def runRev2():
    a1, a2, b1, b2, g1, g2, g3 = 0, 0, 0, 0, 0, 0, 0
    df = pd.read_csv('ratings_musical_Amazon.csv', names=["user", "product", "rating"])
    usersDF = pd.DataFrame(df["user"].unique(), columns=["User"])
    usersDF["Fairness"] = np.nan
    productsDF = pd.DataFrame(df["product"].unique(), columns=["Product"])
    productsDF["Goodness"] = np.nan
    print("Network has %d nodes and %d products" % (len(usersDF.index), len(productsDF.index)))
    iter = 0
    currentgvals = []
    while iter < 500:
        du = 0
        dp = 0
        dr = 0
        print("Epoch %d" % iter)
        print("Goodness")
        median_gvals = productsDF["Goodness"].median()
        for index, row in productsDF.iterrows():
            ftotal = 1.0
            gtotal = df.loc[df['product'] == row["Product"]]["rating"].sum()
            print(gtotal)
            mg = (b1 * median_gvals + gtotal) / (b1 + b2 + ftotal)

            if mg < -1.0:
                x = -1.0
            if mg > 1.0:
                x = 1.0
            dp += abs(row["Goodness"] - mg)
            row["Goodness"] = mg
        print("Fairness")
        iter = 502

def draw_graph(G):
    plt.figure()
    pos = nx.spring_layout(G)
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    nodeShapes = set((aShape[1]["s"] for aShape in G.nodes(data=True)))
    colors = ['green', 'blue']
    for (index, aShape) in enumerate(nodeShapes):
        nx.draw_networkx_nodes(G, pos, node_color=colors[index], node_shape=aShape, nodelist=[sNode[0] for sNode in
                                                                                              filter(lambda x: x[1][
                                                                                                                   "s"] == aShape,
                                                                                                     G.nodes(
                                                                                                         data=True))])

    nx.draw_networkx_edges(G, pos, edgelist=edges, label=weights)

    # nx.draw_networkx(G, pos,with_labels=False, edges=edges,width=weights)
    plt.show()


def createGraph():
    df = pd.read_csv('ratings_musical_Amazon.csv', names=["user", "product", "rating"])
    df = df.head(20)
    G = nx.Graph()
    users = df["user"].unique()
    products = df["product"].unique()
    for u in users:
        G.add_node(u, s="o")
    for p in products:
        G.add_node(p, s="^")

    for (index, row) in df.iterrows():
        G.add_edge(row[0], row[1], weight=row[2])
    draw_graph(G)


if __name__ == '__main__':
    runRev2()

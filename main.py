import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def runRev2():
    a1, a2, b1, b2, g1, g2, g3 = 1, 1, 1, 1, 1, 1, 1
    df = pd.read_csv('ratings_musical_Amazon.csv', names=["user", "product", "rating"])
    df["Fairness"] = np.nan
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
                mg = -1.0
            if mg > 1.0:
                mg = 1.0
            dp += abs(row["Goodness"] - mg)
            row["Goodness"] = mg
        print("Fairness")
        for index, row in df.iterrows():
            rating_distance = 1 - (abs(row["rating"] - productsDF[row["product"]]["Goodness"]) / 2.0)
            user_fairness = usersDF[row["user"]]["Fairness"]
            R = (g2 * rating_distance + g1 * user_fairness) / (g1 + g2 + g3)

            if R < 0.00:
                R = 0.0
            if R > 1.0:
                R = 1.0
            dp += abs(row["Fairness"] - R)
            row["Fairness"] = R

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

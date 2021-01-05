import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt


def runRev2():
    a1, a2, b1, b2, g1, g2, g3 = 1, 1, 1, 1, 1, 1, 1

    df = pd.read_csv('ratings_musical_Amazon.csv', names=["user", "product", "rating"])

    # df = pd.read_csv('amazonWithoutUsersOneTime.csv', names=["user", "product", "rating"])
    # print("Network has %d nodes" % (len(df["user"].index)))
    active = True
    numberUser = 0
    numberProduct = 0
    while active:
        df = df[df.groupby('product').product.transform(len) > 9]
        # df = df[df.groupby('product').product.transform(len) < 6]
        df = df[df.groupby('user').user.transform(len) > 3]
        print("Network has %d nodes and %d products" % (numberUser, numberProduct))

        if numberUser != len(df["user"].unique()) or numberProduct != len(df["product"].unique()):
            numberUser = len(df["user"].unique())
            numberProduct = len(df["product"].unique())
        else:
            active = False

    df.to_csv(r'amazonWithoutUsersOneTime.csv', index=False, header=False)
    print("Network has %d nodes" % (len(df["user"].index)))
    df["Reliability"] = 0

    # convert {1,2,3,4,5} ratings to {-1,-0.5,0, 0.5, 1}
    df["rating"] = (df["rating"] - 3) / 2

    usersDF = pd.DataFrame(df["user"].unique(), columns=["User"])
    usersDF["Fairness"] = 0

    # get only 10 first users
    # usersDF = usersDF[:5000]
    #
    # # list of user ids
    # list_of_users = usersDF["User"].tolist()
    #
    # get ratings only for these specific group of users
    # df = df[df['user'].isin(list_of_users)]

    # create products dataframe
    productsDF = pd.DataFrame(df["product"].unique(), columns=["Product"])
    productsDF["Goodness"] = 0

    print("Network has %d nodes and %d products" % (len(usersDF.index), len(productsDF.index)))

    iter = 0

    while iter < 50:
        du = 0
        dp = 0
        dr = 0
        print("Epoch %d" % iter)

        mf = usersDF['Fairness'].sum() / len(usersDF)
        mg = productsDF['Goodness'].sum() / len(productsDF)

        # GOODNESS
        for index, row in productsDF.iterrows():

            r_up = df.loc[df['product'] == row["Product"]]["Reliability"].tolist()

            score_up = df.loc[df['product'] == row["Product"]]["rating"].tolist()

            sum_up = sum([a * b for a, b in zip(r_up, score_up)])

            in_p = df.loc[df['product'] == row["Product"]]["rating"].count()

            gp = (sum_up + b1 * mg) / (in_p + b1)

            if gp < -1.0:
                gp = -1.0
            if gp > 1.0:
                gp = 1.0

            dp += abs(productsDF.loc[index, 'Goodness'] - gp)
            productsDF.loc[index, 'Goodness'] = gp

        # FAIRNESS
        for index, row in usersDF.iterrows():

            r_up = df.loc[df['user'] == row["User"]]["Reliability"].sum()
            out = df.loc[df['user'] == row["User"]]["Reliability"].count()

            fu = (r_up + a1 * mf) / (out + a1)

            if fu < 0.00:
                fu = 0.0
            if fu > 1.0:
                fu = 1.0
            du += abs(usersDF.loc[index, 'Fairness'] - fu)
            usersDF.loc[index, 'Fairness'] = fu

        # RELIABILITY
        for index, row in df.iterrows():

            fu = float(usersDF.loc[usersDF['User'] == row["user"]]["Fairness"])
            gp = float(productsDF.loc[productsDF['Product'] == row["product"]]["Goodness"])
            score = row["rating"]

            r_up = (g1 * fu + g2 * (1 - abs(score - gp) / 2)) / (g1 + g2)

            if r_up < 0.00:
                r_up = 0.0
            if r_up > 1.0:
                r_up = 1.0
            dr += abs(df.loc[index, 'Reliability'] - r_up)
            df.loc[index, 'Reliability'] = r_up

        iter += 1
        print("Du %f Dp %f Dr %f" % (du, dp, dr))
        if du < 0.01 and dp < 0.01 and dr < 0.01:
            break

    usersDF = usersDF.sort_values(by=["Fairness"], ascending=True)
    productsDF = productsDF.sort_values(by=["Goodness"], ascending=False)
    df = df.sort_values(by=["Reliability"], ascending=False)
    productsDF.to_csv(r'products.csv', index=False, header=True)
    usersDF.to_csv(r'users.csv', index=False, header=True)
    df.to_csv(r'ratings.csv', index=False, header=True)


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
    nx.write_gexf(G, "amazon4_10.gexf")
    # nx.draw_networkx(G, pos,with_labels=False, edges=edges,width=weights)
    plt.show()


def createGraph():
    df = pd.read_csv('amazonWithoutUsersOneTime.csv', names=["user", "product", "rating"])

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
    #runRev2()
    createGraph()

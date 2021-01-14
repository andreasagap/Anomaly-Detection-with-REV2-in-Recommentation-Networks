import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from find_communities import findCommunities


def savePlots(DR, DP, DU, title, iter):
    # Plot Absolute Error
    plt.figure()
    plt.subplot(211)
    plt.subplots_adjust(top=1, bottom=0.5, hspace=1)
    plt.plot(DR, color='red', label='DR')
    plt.plot(DP, color='blue', label='DP')
    plt.plot(DU, color='black', label='DU')
    plt.legend()
    plt.legend(loc=1, prop={'size': 10})
    plt.xlabel('Epochs')
    plt.ylabel('Value')
    plt.title('Absolute Error of DR, DP, DU')

    close = int(iter / 2)
    plt.subplot(212)
    plt.plot(DR[close:], color='red', label='DR')
    plt.plot(DP[close:], color='blue', label='DP')
    plt.plot(DU[close:], color='black', label='DU')
    plt.xticks(np.arange(0, iter - close, step=1), np.arange(close, iter, step=1))
    plt.xlabel('Epochs')
    plt.ylabel('Value')
    plt.title('Zoomed In')
    plt.tight_layout()
    plt.savefig(title + '.png')


def runRev2():

    # smoothing parameters initialization
    params = []
    params.append([0, 0, 0, 1])

    # read Amazon Dataset
    df = pd.read_csv('ratings_musical_Amazon.csv', names=["user", "product", "rating"])
    print("Network has %d nodes" % (len(df["user"].index)))

    # graph sampling based on conditions for reasons of simplicity
    active = True
    numberUser = 0
    numberProduct = 0
    # iterative pruning of nodes
    while active:
        # remove products with less than 10 reviews
        df = df[df.groupby('product').product.transform(len) > 9]
        # remove users with out-degree less than 4
        df = df[df.groupby('user').user.transform(len) > 3]
        if numberUser != len(df["user"].unique()) or numberProduct != len(df["product"].unique()):
            numberUser = len(df["user"].unique())
            numberProduct = len(df["product"].unique())
        else:
            active = False

    # save new graph to csv
    df.to_csv(r'amazonWithoutUsersOneTime.csv', index=False, header=False)
    print("Network has %d nodes" % (len(df["user"].index)))
    df["Reliability"] = 0

    # convert {1,2,3,4,5} ratings to {-1,-0.5,0, 0.5, 1}
    df["rating"] = (df["rating"] - 3) / 2

    # create users DataFrame
    usersDF = pd.DataFrame(df["user"].unique(), columns=["User"])
    usersDF["Fairness"] = 0

    # create products DataFrame
    productsDF = pd.DataFrame(df["product"].unique(), columns=["Product"])
    productsDF["Goodness"] = 0

    print("Network has %d nodes and %d products" % (len(usersDF.index), len(productsDF.index)))

    f = open("results.txt", "a")

    f.write("Network has %d nodes and %d products" % (len(usersDF.index), len(productsDF.index)))
    f.write("\n")

    for p in params:

        # parameter initialization
        a1, b1, g1, g2 = p[0], p[1], p[2], p[3]

        DR = []
        DP = []
        DU = []

        iter = 0
        while iter < 50:
            du = 0
            dp = 0
            dr = 0
            print("Epoch %d" % iter)

            # compute average Fairness & Goodness
            mf = usersDF['Fairness'].sum() / len(usersDF)
            mg = productsDF['Goodness'].sum() / len(productsDF)

            # GOODNESS computation
            for index, row in productsDF.iterrows():

                # rating reliability
                r_up = df.loc[df['product'] == row["Product"]]["Reliability"].tolist()
                # rating score
                score_up = df.loc[df['product'] == row["Product"]]["rating"].tolist()

                # inner product of reliability and rating score vectors
                sum_up = sum([a * b for a, b in zip(r_up, score_up)])

                # in-degree calculation for product nodes
                in_p = df.loc[df['product'] == row["Product"]]["rating"].count()

                # goodness formula
                gp = (sum_up + b1 * mg) / (in_p + b1)

                # normalization for values out of bounds
                if gp < -1.0:
                    gp = -1.0
                if gp > 1.0:
                    gp = 1.0

                # update difference
                dp += abs(productsDF.loc[index, 'Goodness'] - gp)
                # update goodness
                productsDF.loc[index, 'Goodness'] = gp

            # FAIRNESS computation
            for index, row in usersDF.iterrows():

                # rating reliability
                r_up = df.loc[df['user'] == row["User"]]["Reliability"].sum()
                # out-degree of user nodes
                out = df.loc[df['user'] == row["User"]]["Reliability"].count()

                # fairness formula
                fu = (r_up + a1 * mf) / (out + a1)

                # normalization for values out of bounds
                if fu < 0.00:
                    fu = 0.0
                if fu > 1.0:
                    fu = 1.0

                # update differences
                du += abs(usersDF.loc[index, 'Fairness'] - fu)
                # update fairness
                usersDF.loc[index, 'Fairness'] = fu

            # RELIABILITY computation
            for index, row in df.iterrows():

                # get current value for user's fairness
                fu = float(usersDF.loc[usersDF['User'] == row["user"]]["Fairness"])
                # get current value of product's goodness
                gp = float(productsDF.loc[productsDF['Product'] == row["product"]]["Goodness"])
                # get rating score
                score = row["rating"]

                # reliability computation
                r_up = (g1 * fu + g2 * (1 - abs(score - gp) / 2)) / (g1 + g2)

                # normalization for values out of bounds
                if r_up < 0.00:
                    r_up = 0.0
                if r_up > 1.0:
                    r_up = 1.0

                # update differences
                dr += abs(df.loc[index, 'Reliability'] - r_up)
                # update reliability score
                df.loc[index, 'Reliability'] = r_up

            iter += 1

            # save Mean Absolute Error for each iteration
            DR.append(dr)
            DU.append(du)
            DP.append(dp)

            print("Du %f Dp %f Dr %f" % (du, dp, dr))

            # convergence condition
            if du < 0.01 and dp < 0.01 and dr < 0.01:
                break

        # ascending sorting of the results
        usersDF = usersDF.sort_values(by=["Fairness"], ascending=True)
        productsDF = productsDF.sort_values(by=["Goodness"], ascending=False)
        df = df.sort_values(by=["Reliability"], ascending=False)

        # save results to csv
        title = ','.join(str(x) for x in p)
        f.write("Params: " + title)
        f.write("\n")
        f.write("DU: " + ','.join(str(x) for x in DU))
        f.write("\n")
        f.write("DR: " + ','.join(str(x) for x in DR))
        f.write("\n")
        f.write("DP: " + ','.join(str(x) for x in DP))
        f.write("\n")
        f.write("Epoch: " + str(int(iter - 1)))
        f.write("\n-------------------END-------------------\n")

        # save plots
        savePlots(DR, DP, DU, title, iter)

        # create results DataFrame for Goodness, Fairness & Reliability
        productsDF.to_csv(r'products-' + title + '.csv', index=False, header=True)
        usersDF.to_csv(r'users-' + title + '.csv', index=False, header=True)
        df.to_csv(r'ratings-' + title + '.csv', index=False, header=True)

    f.close()


def saveGraphinGexf(G):
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
    saveGraphinGexf(G)


if __name__ == '__main__':
    print("1: Run REV2")
    print("2: Create gexf file")
    print("3: Find community ")
    print("Select 1,2 or 3:")
    option = input()
    if str(option) == "1":
        runRev2()
    elif str(option) == "2":
        createGraph()
    else:
        print("Enter static parameters with comma: ")
        filename = input()
        usersDF = pd.read_csv('users-' + filename + '.csv', names=["User", "Fairness"])
        findCommunities(usersDF)
import networkx as nx
from networkx.algorithms import bipartite
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def runRev2():
    a1, a2, b1, b2, g1, g2, g3 = 1, 1, 1, 1, 1, 1, 1

    df = pd.read_csv('ratings_musical_Amazon.csv', names=["user", "product", "rating"])
    df["Reliability"] = 0
    
    # convert {1,2,3,4,5} ratings to {-1,-0.5,0, 0.5, 1}
    df["rating"] = (df["rating"] - 3)/2

    usersDF = pd.DataFrame(df["user"].unique(), columns=["User"])
    usersDF["Fairness"] = 0

    # get only 10 first users
    usersDF = usersDF[:10]

    # list of user ids
    list_of_users = usersDF["User"].tolist()

    # get ratings only for these specific group of users
    df = df[df['user'].isin(list_of_users)]

    # create products dataframe
    productsDF = pd.DataFrame(df["product"].unique(), columns=["Product"])
    productsDF["Goodness"] = 0

    print("Network has %d nodes and %d products" % (len(usersDF.index), len(productsDF.index)))

    iter = 0

    while iter < 10:

        print("Epoch %d" % iter)

        mf = usersDF['Fairness'].sum() / len(usersDF)
        mg = productsDF['Goodness'].sum() / len(productsDF)

        # GOODNESS
        for index, row in productsDF.iterrows():

            r_up = df.loc[df['product'] == row["Product"]]["Reliability"].tolist()

            score_up = df.loc[df['product'] == row["Product"]]["rating"].tolist()

            sum_up = sum([a * b for a, b in zip(r_up, score_up)])

            in_p = df.loc[df['product'] == row["Product"]]["rating"].count()

            gp = (sum_up + b1*mg) / (in_p + b1)

            if gp < -1.0:
                gp = -1.0
            if gp > 1.0:
                gp = 1.0

            productsDF.loc[index, 'Goodness'] = gp
        print(productsDF)

        # FAIRNESS
        for index, row in usersDF.iterrows():

            r_up = df.loc[df['user'] == row["User"]]["Reliability"].sum()
            out = df.loc[df['user'] == row["User"]]["Reliability"].count()

            fu = (r_up + a1*mf) / (out + a1)

            if fu < 0.00:
                fu = 0.0
            if fu > 1.0:
                fu = 1.0

            usersDF.loc[index, 'Fairness'] = fu
        print(usersDF)

        # RELIABILITY
        for index, row in df.iterrows():

            fu = float(usersDF.loc[usersDF['User'] == row["user"]]["Fairness"])
            gp = float(productsDF.loc[productsDF['Product'] == row["product"]]["Goodness"])
            score = row["rating"]

            r_up = (g1*fu + g2*(1 - abs(score-gp)/2)) / (g1+g2)

            if r_up < 0.00:
                r_up = 0.0
            if r_up > 1.0:
                r_up = 1.0

            df.loc[index, 'Reliability'] = r_up

        print(df)

        iter += 1


if __name__ == '__main__':

    runRev2()

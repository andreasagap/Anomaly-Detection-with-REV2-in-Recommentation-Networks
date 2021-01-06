import networkx as nx
import pandas as pd

import matplotlib.pyplot as plt
plt.rcParams.update(plt.rcParamsDefault)
plt.rcParams.update({'figure.figsize': (15, 10)})

import networkx.algorithms.community as nxcom

from networkx.algorithms import bipartite

df = pd.read_csv('amazonWithoutUsersOneTime.csv', names=["user", "product", "rating"])

# Select 10 users with the lowest fairness and collect their reviews
usersDF = usersDF.head(10)
users = usersDF['User'].tolist()
df2 = pd.DataFrame(columns = df.columns)
for i in range(10):
    df2 = df2.append(df[df['user'] == users[i]])

G = nx.Graph()
users = df2["user"].unique()
products = df2["product"].unique()
for u in users:
    G.add_node(u, s="o", bipartite=0)
for p in products:
    G.add_node(p, s="^", bipartite=1)

for (index, row) in df2.iterrows():
    G.add_edge(row[0], row[1], weight=row[2])
    
def set_node_community(G, communities):
    # Add community to node attributes
    for c, v_c in enumerate(communities):
        for v in v_c:
            # Add 1 to save 0 for external edges
            G.nodes[v]['community'] = c + 1

def set_edge_community(G):
    # Find internal edges and add their community to their attributes
    for v, w, in G.edges:
        if G.nodes[v]['community'] == G.nodes[w]['community']:
            # Internal edge, mark with community
            G.edges[v, w]['community'] = G.nodes[v]['community']
        else:
            # External edge, mark as 0
            G.edges[v, w]['community'] = 0

def get_color(i, r_off=1, g_off=1, b_off=1):
    # Assign a color to a vertex.'''
    r0, g0, b0 = 0, 0, 0
    n = 16
    low, high = 0.1, 0.9
    span = high - low
    r = low + span * (((i + r_off) * 3) % n) / (n - 1)
    g = low + span * (((i + g_off) * 5) % n) / (n - 1)
    b = low + span * (((i + b_off) * 7) % n) / (n - 1)
    return (r, g, b)      

### Find Communities in Users_set
    
# Distinguish between two sets
top_nodes = {n for n, d in G.nodes(data=True) if d["bipartite"] == 0}
bottom_nodes = set(G) - top_nodes

# Get User set
U = bipartite.projected_graph(G, top_nodes)

# Find communities based on modularity
communities = sorted(nxcom.greedy_modularity_communities(U), key=len, reverse=True)

set_node_community(U, communities)
set_edge_community(U) 

node_color = [get_color(U.nodes[v]['community']) for v in U.nodes]

external = [(v, w) for v, w in U.edges if U.edges[v, w]['community'] == 0]
internal = [(v, w) for v, w in U.edges if U.edges[v, w]['community'] > 0]
internal_color = ['black' for e in internal]

users_pos = nx.spring_layout(U)

plt.rcParams.update({'figure.figsize': (15, 10)})

nx.draw_networkx(
        U,
        pos=users_pos,
        node_size=0,
        edgelist=external,
        edge_color="silver",
        with_labels=False)

nx.draw_networkx(
        U,
        pos=users_pos,
        node_color=node_color,
        edgelist=internal,
        edge_color=internal_color,
        with_labels=False)

plt.show()

# Select biggest community
biggest_community = communities[0]

# Exctract users from biggest community
clique_users_list = []
for x in biggest_community:
    clique_users_list.append(x)

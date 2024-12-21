import networkx as nx 

from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt


"""
Creates a matrix representing how much topics from W are correlated

Args: (m, p) matrix W, rows representing sentences, columns representing topics
Returns: Weight matrix
"""
def weight_matrix(W):
    return cosine_similarity(W.T)

"""
shows a undirected weighted graph.

Args: 
    - weights: weight matrix
    - labels: labels for the nodes
    - lam: minimum threshold of weights to show in the graph
Returns: 
"""
def visualize_graph(weights, labels, lam):
    G = nx.Graph()
    n = len(weights)

    for i in range(n):
        for j in range(i+1, n):
            w = weights[i][j]
            if w > lam or lam == -1:
                G.add_edge(i, j, weight=w)
    pos = nx.spring_layout(G)

    edges = G.edges(data=True)
    wts = [d['weight'] for (_,_,d) in edges]
    nx.draw(G, pos, with_labels=False, edge_color=wts, edge_cmap=plt.cm.Blues, node_size=1600)
    node_labels = {i: "\n".join(labels[i]) for i in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10)
    plt.show()

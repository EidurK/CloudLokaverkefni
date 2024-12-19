import numpy as np
import math
import networkx as nx 

from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

def adjacency_matrix(row):
    return np.outer(row, row)

def weight_matrix_entry(row):
    sum_matrix = (row[:, None] + row[None, :]) / 2
    return sum_matrix

def sum_normalize(A, v):
    return (A - A.min()) / (A.max() - A.min())

# def weight_matrix(data):
#     p = data.shape[1]
#     final = np.zeros((p,p))
#     for row in data:
#         final += weight_matrix_entry(row)
#     final = sum_normalize(final,np.sum(data, axis=0))
#     return final


def weight_matrix(W):
    return cosine_similarity(W.T)

# def visualize_graph(weights, lam):
#     G = nx.Graph()
#     n = len(weights)
#     for i in range(n):
#         for j in range(i+1, n):
#             w = weights[i][j]
#             if w > lam or lam==-1:
#                 G.add_edge(i, j, weight=w)
#     pos = nx.spring_layout(G)
#
#     edges = G.edges(data=True)
#     wts = [d['weight'] for (_,_,d) in edges]
#     nx.draw(G, pos, with_labels=True, edge_color=wts, edge_cmap=plt.cm.Blues)
#     plt.show()
#     G = nx.Graph()
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

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF


"""
Factorizes (m, n) matrix X into (m, p) and (p, n) matrices W and H

Args: numpy_matrix X, int p
Returns: W and H
"""
def nmf(X, p):
    model = NMF(n_components=p, init='random', random_state=0)
    W = model.fit_transform(X)
    H = model.components_
    return W,H


"""
Creates a TF-IDF matrix and corresponding vocabulary from a list of Documents

Args: 
    - D: list of Documents 
    - min_df: minimum document frequency
    - max_df: maximum document frequency
    - max_features: maximum number of features in vocabulary
Returns: TF-IDF matrix X and corresponding vocabulary vocab
"""
def tfid_vectorizer(D, min_df = 0.05, max_df=0.6, max_features=1000):
    vectorizer = TfidfVectorizer(
            min_df = min_df,
            max_df=max_df,
            max_features=max_features,
            lowercase=False)
    X = vectorizer.fit_transform(D)
    vocab = vectorizer.get_feature_names_out()
    return X, vocab

"""
Finds the indexes of the higest n values for each row in a Matrix

Args: 
    - H: Matrix 
    - n: int 
Returns: matrix of the higest n values for each row in the matrix
"""
def top_n_index(H, n):
    return [np.argsort(h[:])[-n:] for h in H]


"""
Finds the top words for each topic in H matrix (from NMF)

Args: 
    - H: Matrix 
    - n: int 
    - vocab: vocabulary
Returns: matrix of top words for each topic
"""
def top_n_words(H,n,vocab):
    indx = top_n_index(H,n)
    return [vocab[i] for i in indx]

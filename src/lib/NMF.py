import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

def nmf(X, p):
    model = NMF(n_components=p, init='random', random_state=0)
    W = model.fit_transform(X)
    H = model.components_
    return W,H

def tfid_vectorizer(D, min_df = 0.05, max_df=0.6, max_features=1000):
    vectorizer = TfidfVectorizer(
            min_df = min_df,
            max_df=max_df,
            max_features=max_features,
            lowercase=False)
    X = vectorizer.fit_transform(D)
    vocab = vectorizer.get_feature_names_out()
    return X, vocab

def top_n_index(H, n):
    return [np.argsort(h[:])[-n:] for h in H]

def top_n_words(H,n,vocab):
    indx = top_n_index(H,n)
    return [vocab[i] for i in indx]

def top_n_sentences(W, n, D):
    indx = top_n_index(W.T, n)
    return [D[i] for i in indx]

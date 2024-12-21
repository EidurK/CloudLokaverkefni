import sys
import os

sys.path.append(os.path.abspath("./lib"))

import lib.NMF as NMF
from pathlib import Path
import pandas as pd
import lib.weighted_graph as wg
import numpy as np

# ------------------------------

input_file = Path("../data/FullyCleanedDataframe.csv")

min_df = 0.01
max_df = 0.1
max_features = 1000

topic_count = 65
word_count_per_group = 6

# ------------------------------

if not input_file.exists():
    print('file not exist')

df = pd.read_csv(input_file, encoding='utf-8')
df = df.dropna()

D = df['text'].to_numpy()

X, vocab = NMF.tfid_vectorizer(D,min_df,max_df, max_features=max_features)
W,H = NMF.nmf(X,topic_count)

top_words = NMF.top_n_words(H, word_count_per_group, vocab)

label_count = 3 # for graph visualization
labels = []
for group in top_words:
    g = np.flip(group)
    print(g)
    labels.append(g[-label_count:])
    
matrix = wg.weight_matrix(W)

while True:
    lam = float(input("input weight threshold between 0-1 or -1 to quit:"))
    if lam == -1:
        break
    wg.visualize_graph(matrix, labels, lam)

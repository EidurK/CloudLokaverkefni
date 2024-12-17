import sys
import os

sys.path.append(os.path.abspath("./lib"))

import lib.NMF as NMF
from pathlib import Path
import pandas as pd

# ------------------------------
input_file = Path("../data/FullyCleanedDataframe.csv")

min_df = 0.01
max_df = 0.1
max_features = 1000

topic_count = 10
word_count_per_group = 8


# ------------------------------
if not input_file.exists():
    print('file not exist')

df = pd.read_csv(input_file, encoding='utf-8')
df = df.dropna()
D = df['text'].to_numpy()
X, vocab = NMF.tfid_vectorizer(D,min_df,max_df, max_features=max_features)
print("vocabulary size:", vocab.shape[0])
W,H = NMF.nmf(X,topic_count)

top_words = NMF.top_n_words(H,word_count_per_group, vocab)

for group in top_words:
    print(group)

top_sentences = NMF.top_n_sentences(W,1,D)
groups = []
for i in groups:
    print(top_sentences[i])

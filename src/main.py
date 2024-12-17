import sys
import os

sys.path.append(os.path.abspath("./lib"))

import NMF
from pathlib import Path
import pandas as pd

# ------------------------------
my_file = Path("../data/FullyCleanedDataframe.csv")

min_df = 0.05
max_df = 0.2
max_features = 500

topic_count = 15
word_count_per_group = 8


# ------------------------------
if not my_file.exists():
    print('file not exist')

df = pd.read_csv(my_file)
df = df.dropna()
D = df['text'].to_numpy()
X, vocab = NMF.tfid_vectorizer(D,min_df,max_df, max_features=max_features)
W,H = NMF.nmf(X,topic_count)

top_words = NMF.top_n_words(H,word_count_per_group, vocab)

for group in top_words:
    print(group)

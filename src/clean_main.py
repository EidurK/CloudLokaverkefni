import sys
import os

sys.path.append(os.path.abspath("./lib"))
import pandas as pd
import clean_data as cleaning

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Ekki breyta neinu (helst)
input_file='../data/PostsOfQAnon.csv'
output_file = '../data/FullyCleanedDataframe.csv' 

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

df = pd.read_csv(input_file)
print(df.head(3))
df = cleaning.clean_dataframe(df)
print(df.head(3))

df.to_csv(output_file)

import numpy as np
import pandas as pd
import re
import nltk

from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet, stopwords

"""
Cleans the input string by:

Removing all non-letter characters [A-z].
Removing double spaces.
Stripping spaces at the start and end.
Converting all characters to lowercase.
"""
def remove_symbols(string): 
    clean_string = re.sub(r'[^A-Za-z\s]', '', string)  # Keep letters and spaces
    clean_string = re.sub(r'\s+', ' ', clean_string)  # Replace multiple spaces with one
    clean_string = clean_string.strip().lower()       # Strip and convert to lowercase
    return clean_string

"""
Removes all words starting with 'http' up to the next space.

Args: str - Input string containing links.
Returns: str - Text with links removed.
"""
def remove_links(text):
    cleaned_text = re.sub(r'http\S+', '', text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    return cleaned_text.strip()


def stopwords_lemmatizer(text, stopwords, testing=False):
    if testing:
        nltk.download('stopwords')
        nltk.download('wordnet')


    tokenized_list =  [word for word in text.split() if word.lower() not in stopwords]

    wordnet_lemmatizer = WordNetLemmatizer()
    snowball_stemmer = SnowballStemmer('english')

    lemmatized_words = []
    for word in tokenized_list:
        lemmatized_words.append(wordnet_lemmatizer.lemmatize(word))

    cleaned_list  = []
    for word in lemmatized_words:
        cleaned_list.append(snowball_stemmer.stem(word))
    return ' '.join(cleaned_list)

def create_stopwords():
    stopwords = np.array(nltk.corpus.stopwords.words('english'))
    
    other_half =np.array(open("../data/stop_words.txt").readlines())
    stopwords = np.concatenate((stopwords, other_half), axis=None)

    return stopwords


def clean_row(string): 
    if not isinstance(string, str):
        return string  # Handle non-string inputs gracefully

    stopwords = create_stopwords()
    string = stopwords_lemmatizer(string, stopwords)
    string = remove_links(string) 
    string = remove_symbols(string)
    
    return string

def clean_dataframe(df):
    nltk.download('stopwords')
    nltk.download('wordnet')

    df['text'] = df['text'].apply(clean_row)
    return df



import numpy as np
import pandas as pd
import re
import nltk

from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer


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


"""
Removes all stopwords and lemmatizes the input string

Args: str, list, bool - input string, list of stopwords.
Returns: str - Cleaned string.
"""
def stopwords_lemmatizer(text, stopwords):
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

"""
Creates a list of stopwords form nltk's stopwords and the stopwords from ./data/stop_words.txt

Returns: list - list of stopwords.
"""
def create_stopwords():
    stopwords = np.array(nltk.corpus.stopwords.words('english'))
    
    other_half = np.array([line.strip() for line in open("../data/stop_words.txt", encoding="utf-8").readlines()])

    stopwords = np.concatenate((stopwords, other_half), axis=None)

    return stopwords



"""
A pipeline used to clean text

Args: str, list - a row of a dataframe containing text
Returns: str - cleaned text
"""
def clean_row(string, stopwords): 
    if not isinstance(string, str):
        return string  # Handle non-string inputs gracefully

    string = stopwords_lemmatizer(string, stopwords)
    string = remove_links(string) 
    string = remove_symbols(string)
    
    return string


"""
Cleans the text column of a given dataframe.

Args: dataframe with column 'text'.
Returns: dataframe with the 'text' column cleaned.
"""
def clean_dataframe(df):
    nltk.download('stopwords')
    nltk.download('wordnet')
    stopwords = create_stopwords()
    
    df['text'] = df['text'].apply(lambda string: clean_row(string, stopwords))
    return df



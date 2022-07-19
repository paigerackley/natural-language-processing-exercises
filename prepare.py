import unicodedata
import re
import json

import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords

import pandas as pd

import acquire

import warnings
warnings.filterwarnings('ignore')



def basic_clean(string):
    '''
    This function takes in a string and
    returns the string normalized.
    '''
    string = unicodedata.normalize('NFKD', string)\
             .encode('ascii', 'ignore')\
             .decode('utf-8', 'ignore')
    string = re.sub(r'[^\w\s]', '', string).lower()
    return string

def stem(string):
    ps = nltk.porter.PorterStemmer()
    # Use the stemmer to stem each word in the list of words we created by using split.
    stems = [ps.stem(word) for word in string.split()]
    # Join our lists of words into a string again and assign to a variable.
    string = ' '.join(stems)
    return string



def lemmatize(string):
    wnl = nltk.stem.WordNetLemmatizer()
    # Use the lemmatizer on each word in the list of words we created by using split.
    lemmas = [wnl.lemmatize(word) for word in string.split()]
    # Join our list of words into a string again and assign to a variable.
    string = ' '.join(lemmas)
    return string


def remove_stopwords(string, extra_words = [], exclude_words = []):
        stopword_list = stopwords.words('english')
        # Remove 'exclude_words' from stopword_list to keep these in my text.
        stopword_list = set(stopword_list) - set(exclude_words)
        # Add in 'extra_words' to stopword_list.
        stopword_list = stopword_list.union(set(extra_words))
        # Split words in string.
        words = string.split()
        # Create a list of words from my string with stopwords removed and assign to variable.
        filtered_words = [word for word in words if word not in stopword_list]
        # Join words in the list back into strings and assign to a variable.
        string_without_stopwords = ' '.join(filtered_words)
        return string_without_stopwords




## FINAL PREPARE ##



#creating function of above needed items:
def prep_article_data(df, column, extra_words=[], exclude_words=[]):
    '''
    This function takes in a df and the string name for a text column with the
    option to pass lists for extra_words and exclude_words and
    returns a df with the text article title, original text, stemmed text,
    lemmatized text, cleaned, tokenized, & lemmatized text with stopwords removed.
    '''
    df['clean'] = df[column].apply(basic_clean)\
                            .apply(tokenize)\
                            .apply(remove_stopwords,
                                  extra_words=extra_words,
                                  exclude_words=exclude_words)
    
    df['stemmed'] = df['clean'].apply(stem)
    
    df['lemmatized'] = df['clean'].apply(lemmatize)
    
    return df[['title', column,'clean', 'stemmed', 'lemmatized']]
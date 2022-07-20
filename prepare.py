import unicodedata
import re
import json

import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords

import pandas as pd

import acquire

def basic_clean(string):
    
    # lowercase everything
    string = string.lower()
    
    # remove inconsistenceis
    # encode into ascii byte strings
    # decode back into UTF-8
    # (This process will normalize the unicode characters)
    
    string = unicodedata.normalize('NFKD', string).encode('ascii', 'ignore').decode('UTF-8')
    
    # replace anything that is not a letter, number, whitespace, etc
    # use regex to perform this operation
    string = re.sub(r"[^a-z0-9\s]", '', string)
    
    return string



def tokenize(string):
    """
    This function will take in a string, tokenize the string and 
    return the tokenize string
    """
    
    #create the token
    token = nltk.tokenize.ToktokTokenizer()
    
    #Use the token
    string = token.tokenize(string,  return_str=True)
    
    return string

def stem(string):
    """
    This function will accept some text(string) and return a stemmed 
    version of the text
    """
    
    #create the porter stem
    ps = nltk.porter.PorterStemmer()
    
    #Apply the stem to each work in the string and create a list
    # of steemed words
    
    stem = [ps.stem(word) for word in string.split()]
    
    # rejoin the string together
    stemmed_string = ' '.join(stem)
    
    return stemmed_string

def lemmatize(string):
    """This function takes in a string and returns a lmeeatized 
    version of the string"""
    
    # create the lemmatizer
    wnl = nltk.stem.WordNetLemmatizer()
    
    lemmas = [wnl.lemmatize(word) for word in string.split()]
    
    string_lemmatize = ' '.join(lemmas)
    
    return string_lemmatize


def remove_stopwords(string, extra_words=[], exclude_words=[]):
    
    #get english stopwords from nltk
    stop_words = stopwords.words('english')
    
    #Add extra words to be removed to the stop word list
    for word in extra_words:
        stop_words.append(word)
    
    #Remove words to be excluded from the stop word list
    for word in exclude_words:
        stop_words.remove(word)
    
    #Create a list of words to be checked by splitting the string
    words = string.split()
    
    #Filter out all of the stop words
    filtered_words = [word for word in words if word not in stop_words]
    
    #Join the list of filtered words into a string
    filtered_string = ' '.join(filtered_words)
    
    return filtered_string




#################################################################################


def news_df_prepare():
    news_df = acquire.get_all_inshorts_articles()
    news_df.drop(columns = 'category', inplace=True)
    news_df.rename({'content':'original'}, axis=1, inplace=True)
    news_df['clean'] = news_df.original.apply(basic_clean)
    news_df['clean'] = news_df.clean.apply(tokenize)
    news_df['clean'] = news_df.clean.apply(remove_stopwords)
    news_df['stemmed'] = news_df.clean.apply(stem)
    news_df['lemmatized'] = news_df.clean.apply(lemmatize)
    return news_df


def codeup_df_prepare():
    codeup_df = acquire.get_all_codeup_blogs()
    codeup_df.drop(columns= ['date','link'], inplace=True)
    codeup_df.rename({'content':'original'}, axis=1, inplace=True)
    codeup_df['clean'] = codeup_df.original.apply(basic_clean)
    codeup_df['clean'] = codeup_df.clean.apply(tokenize)
    codeup_df['clean'] = codeup_df.clean.apply(remove_stopwords)
    codeup_df['stemmed'] = codeup_df.clean.apply(stem)
    codeup_df['lemmatized'] = codeup_df.clean.apply(lemmatize)
    return codeup_df


##############################################################################

def prep_text(df, column, extra_words=[], exclude_words=[]):
    
    df['clean'] = df[column].apply(basic_clean)\
                            .apply(tokenize)\
                            .apply(remove_stopwords,
                                  extra_words=extra_words,
                                  exclude_words=exclude_words)
    
    df['stemmed'] = df['clean'].apply(stem)
    
    df['lemmatized'] = df['clean'].apply(lemmatize)
    
    return df[['label', column,'clean', 'stemmed', 'lemmatized']]
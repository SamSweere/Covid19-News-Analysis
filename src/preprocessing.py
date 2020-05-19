import read_data
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from itertools import chain
import re
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords
import pandas as pd
import seaborn as sns

"""
Note: for now no sklearn pipeline bc of overhead
"""

def tokenize_sentences(df, colname="body"):
    """ sentence tokenization """
    df["sentences"] = df[colname].apply(nltk.sent_tokenize)
    return df

def tokenize_words(df):
    """ word tokenization """
    df["token"] = df["sentences"].apply(lambda x: [nltk.word_tokenize(sentence) for sentence in x])
    return df

def tag_POS(df):
    """ POS tagging """
    df["POS_token"] = df["token"].apply(lambda x: [nltk.pos_tag(sentence) for sentence in x])
    return df

# lemmatization
def get_wordnet_pos(treebank_tag):
    """ helper function """
    if treebank_tag.startswith("J"):
        return wordnet.ADJ
    elif treebank_tag.startswith("V"):
        return wordnet.VERB
    elif treebank_tag.startswith("N"):
        return wordnet.NOUN
    elif treebank_tag.startswith("R"):
        return wordnet.ADV
    elif treebank_tag.startswith("S"):
        return wordnet.ADJ_SAT
    else:
        return ""

def lemmatize_sentence(pos_token_list, lemmatizer):
    """
    helper function
    @param pos_token_list: list of POS token
    """
    result = []
    for word in pos_token_list:
        if get_wordnet_pos(word[1]) != "":
            result.append(lemmatizer.lemmatize(word[0], get_wordnet_pos(word[1])))
        else:
            result.append(word[0])
    return result
        
def lemmatize(df):
    """ lemmatize """
    lemmatizer = WordNetLemmatizer()
    df["lemmas"] = df["POS_token"].apply(
        lambda x: [lemmatize_sentence(sentence, lemmatizer) for sentence in x])
    return df

def stop(token_list, custom_stopwords):
    """ helper function """
    res = []
    for token in token_list:
        if token.isalpha() and (1 < len(token)) and (token.lower() not in custom_stopwords):
            res.append(token.lower())
    return res

def remove_stopwords(df):
    """ remove stopwords """
    custom_stopwords = stopwords.words("english")
    df["token"] = df["lemmas"].apply(lambda x: list(chain.from_iterable(x)))
    df["token"] = df["token"].apply(lambda x: stop(x, custom_stopwords))
    return df

# Main function
def preprocess(df, colname="body"):
    """ 
    Pipeline 
    @param df: DataFrame with column of strings containing articles
    @param colname: Name of columns containing articles
    """
    if len(set(["sentences", "token", "POS_token", "lemma"]).intersection(set(df.columns))) != 0:
        print("DataFrame contains columns that will be overwritten")
        raise NameError
    df = tokenize_sentences(df, colname)
    df = tokenize_words(df)
    df = tag_POS(df)
    df = lemmatize(df)
    df = remove_stopwords(df)
    return df
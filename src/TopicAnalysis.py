"""
0) Read in as pandas df
1) Tokenization
2) Part-of-speech tagging
3) Syntacitic parsing
4) Lemmatization & Stemming
5) Stopword removal
5) Vectorization
"""

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
import preprocessing
from gensim.models import Phrases
import numpy as np
from gensim import models
from gensim import corpora
from datetime import datetime

"""GOAL: Topic detection"""

# TODO apparently people don't do stemming anymore these days
# TODO it's probably better if we let spacy do this

### Vectorization

def build_dictionary(token_list):
    """ build and filter dictionary from tokens """
    # build a dictionary from our tokens
    dictionary_LDA = corpora.Dictionary(token_list)
    # threshold for rare words
    dictionary_LDA.filter_extremes(no_below=3)
    return dictionary_LDA


def lda(token_list, dictionary_LDA, num_topics):
    """ Run Latent Dirichlet Allocation """
    # convert document to bag of words
    corpus = [dictionary_LDA.doc2bow(t) for t in token_list]
    np.random.seed(123456)
    lda_model = models.LdaModel(corpus, num_topics=num_topics, \
                                id2word=dictionary_LDA, \
                                passes=4, alpha=[0.01]*num_topics, \
                                eta=[0.01]*len(dictionary_LDA.keys()))
    return lda_model, corpus

def n_gram(df):
    """ 
    Trigram model of word probabilities (old-fashioned TM)
    @param df: DataFrame with a 'token' column
    """
    token = df.token.tolist()
    bigram_model = Phrases(token)
    trigram_model = Phrases(bigram_model[token], min_count=1)
    token_list = list(trigram_model[bigram_model[token]])
    return token_list


### Topic Analysis

def show_topics(lda_model, num_topics):
    """ helper function """
    for i, topic in lda_model.show_topics(formatted=True, num_topics=num_topics, num_words=20):
        print(str(i) + ": " + topic[0:60] + "...")

def predict_topic(lda_model, dictionary_LDA, num_topics, document_list):
    """ Predicting topics on unseen documents """
    # TODO build df from list of tuples?
    doc_id = []
    topic = []
    weight = []
    words_in_topic = []
    topics = lda_model.show_topics(formatted=True, num_topics=num_topics, num_words=20)
    for d_id, doc in enumerate(document_list):
        tokens = nltk.word_tokenize(doc)
        for i in lda_model[dictionary_LDA.doc2bow(tokens)]:  # choose highest-ranked only
            doc_id.append(d_id)
            topic.append(i[0])
            weight.append(i[1])
            words_in_topic.append(topics[i[0]][1])
    topic_df = pd.DataFrame({
        "doc_id": doc_id,
        "topic": topic,
        "weight": weight,
        "words_in_topic": words_in_topic
    })
    return topic_df



if __name__ == "__main__":
    # Inspiration https://github.com/FelixChop/MediumArticles/blob/master/LDA-BBC.ipynb
    # TODO what is this Chandigarh stuff in body?

    # TODO it might make sense to somehow pre-allocate the data well
    ### Topic Analysis

    # load data
    start_date = datetime.strptime("2020-03-10", "%Y-%m-%d")
    end_date = datetime.strptime("2020-04-20", "%Y-%m-%d")
    df = read_data.get_body_df(
        n_articles=1000, start_date=start_date, end_date=end_date)

    df = preprocessing.preprocess(df)
    n_gram_list = n_gram(df)
    num_topics = 10

    # LDA results
    dictionary_LDA = build_dictionary(n_gram_list)
    lda_model, corpus = lda(n_gram_list, dictionary_LDA, num_topics)
    show_topics(lda_model, num_topics)  # show some intermediate results

    # predict (TODO should we preprocess in same way as text?)
    doc = read_data.get_body_df(15).body[10]
    topic_df = predict_topic(lda_model, dictionary_LDA, num_topics, [doc])
    print(topic_df)


    # Visualize results
    import pyLDAvis.gensim
    lda_display3 = pyLDAvis.gensim.prepare(
        lda_model, corpus, dictionary_LDA, sort_topics=False)
    pyLDAvis.show(lda_display3)



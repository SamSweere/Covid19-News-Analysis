"""
0) Read in as pandas df
1) Tokenization
2) Part-of-speech tagging
3) Syntacitic parsing
4) Lemmatization & Stemming
5) Stopword removal
5) Vectorization
"""

"""GOAL: Topic detection"""

# Inspiration https://github.com/FelixChop/MediumArticles/blob/master/LDA-BBC.ipynb

# TODO what is this Chandigarh stuff in body?

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

# read in a given number of articles
df = read_data.get_body_df(n_articles=10)

# sentence tokenization
df["sentences"] = df["body"].apply(nltk.sent_tokenize)

# word tokenization
df["token"] = df["sentences"].apply(lambda x: [nltk.word_tokenize(sentence) for sentence in x])

# POS tagging
df["POS_token"] = df["token"].apply(lambda x: [nltk.pos_tag(sentence) for sentence in x])

# lemmatization
def get_wordnet_pos(treebank_tag):
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

def lemmatize(pos_token_list):
    """@param pos_token_list: list of POS token"""
    result = []
    for word in pos_token_list:
        if get_wordnet_pos(word[1]) != "":
            result.append(lemmatizer.lemmatize(word[0], get_wordnet_pos(word[1])))
        else:
            result.append(word[0])
    return result
        

lemmatizer = WordNetLemmatizer()
df["lemmas"] = df["POS_token"].apply(lambda x: [lemmatize(sentence) for sentence in x])

# remove stopwords
custom_stopwords = stopwords.words("english")

def upper_lower(token_list):
    res = []
    for token in token_list:
        if token.isalpha() and (1 < len(token)) and (token.lower() not in custom_stopwords):
            res.append(token.lower())
    return res

df["token"] = df["lemmas"].apply(lambda x: list(chain.from_iterable(x)))
df["token"] = df["token"].apply(upper_lower)


### LDA (Latent Dirichlet allocation)

from gensim.models import Phrases

# TODO what's going on here
token = df.token.tolist()
bigram_model = Phrases(token)
trigram_model = Phrases(bigram_model[token], min_count=1)
token = list(trigram_model[bigram_model[token]])

# Run LDA
from gensim import models
from gensim import corpora
import numpy as np

dictionary_LDA = corpora.Dictionary(token)
dictionary_LDA.filter_extremes(no_below=3)
corpus = [dictionary_LDA.doc2bow(t) for t in token]

np.random.seed(123456)
num_topics = 3
lda_model = models.LdaModel(corpus, num_topics=num_topics, \
                            id2word=dictionary_LDA, \
                            passes=4, alpha=[0.01]*num_topics, \
                            eta=[0.01]*len(dictionary_LDA.keys()))

### Exploration of LDA results


# Looks pretty good!
for i, topic in lda_model.show_topics(
    formatted=True, num_topics=num_topics, num_words=20):
    print(str(i)+": "+ topic)
    print()

# Topics to data frame

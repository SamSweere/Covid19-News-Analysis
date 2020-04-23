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
import TopicAnalysis
from datetime import datetime

# Inspiration https://github.com/FelixChop/MediumArticles/blob/master/LDA-BBC.ipynb
# TODO what is this Chandigarh stuff in body?

# TODO it might make sense to somehow pre-allocate the data well

# load data
start_date = datetime.strptime("2020-03-10", "%Y-%m-%d")
end_date = datetime.strptime("2020-04-20", "%Y-%m-%d")
df = read_data.get_body_df(
    n_articles=1000, start_date=start_date, end_date=end_date)

df = preprocessing.preprocess(df)
n_gram_list = TopicAnalysis.n_gram(df)
num_topics = 10

# LDA results
dictionary_LDA = TopicAnalysis.build_dictionary(n_gram_list)
lda_model, corpus = TopicAnalysis.lda(n_gram_list, dictionary_LDA, num_topics)
TopicAnalysis.show_topics(lda_model, num_topics)  # show some intermediate results

# predict (TODO should we preprocess in same way as text?)
doc = read_data.get_body_df(15).body[10]
topic_df = TopicAnalysis.predict_topic(lda_model, dictionary_LDA, num_topics, [doc])
print(topic_df)


IsADirectoryError
# Visualize results

import pyLDAvis.gensim
lda_display3 = pyLDAvis.gensim.prepare(
    lda_model, corpus, dictionary_LDA, sort_topics=False)
pyLDAvis.show(lda_display3)

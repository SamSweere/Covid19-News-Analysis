import sys
sys.path.append("src/")
import spacy
from sklearn.decomposition import NMF
import read_data
from datetime import datetime
import os
import textacy.corpus
from textacy.vsm import Vectorizer
import textacy.tm
import re
import string
import numpy as np
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import scipy
import pandas as pd


# TODO should probably just inherit from model or something...
class TopicAnalyser:
    def __init__(self):
        self.vectorizer = self.get_vectorizer()
        self.topic_model = self.get_nmf_model()
        self.graphicspath = "src/figures/"
        self.image_type = ".png"

    def identify_topics(self):
        """ Find topics in data """
        pass

    def label_data(self):
        """ Label data with known topicy """
        pass

    def apply_nlp(self, df):
        # remove punctuation
        df["clean_body"] = df["body"].apply(lambda x: self.clean_text(x))
        # We rely on spacy for all preprocessing
        # TODO not sure if we are feeding textacy exactly what it wants
        model_size = "sm"
        nlp = spacy.load(f"en_core_web_{model_size}", disable=["parser", "ner"])
        df["nlp"] = df["clean_body"].apply(nlp)
        return df

    def get_vectorizer(self):
        vectorizer = Vectorizer(
            tf_type="linear",
            apply_idf=True,
        )
        return vectorizer

    def get_nmf_model(self):
        # TODO how can we get this to converge?
        model = textacy.tm.TopicModel("nmf", n_topics=10)
        return model

    def get_doc_term_matrix(self, df, fit=False):
        """ 
        Get doc_term_matrix for docs in df. feed docs as nested list; maintains order so we can just keep using our df
        """
        # print(f"Weighting formula: {self.vectorizer.weighting}")  # tfidf
        docs = [i.doc.text for i in df["nlp"]]
        my_terms_list=[[tok  for tok in doc.split() if tok not in stopwords.words('english') ] for doc in docs]
        if fit:
            self.vectorizer.fit(my_terms_list)
        doc_term_matrix = self.vectorizer.fit_transform(my_terms_list)
        # print("Some Terms:")
        # print(self.vectorizer.terms_list[:10])

        return doc_term_matrix

    def get_doc_topic_matrix(self, doc_term_matrix, fit=False):
        if fit:
            self.topic_model.fit(doc_term_matrix)
        doc_topic_matrix = self.topic_model.transform(doc_term_matrix)
        print(f"Doc topic matrix shape: {doc_topic_matrix.shape}")
        return doc_topic_matrix

    def visualize(self, title, doc_term_matrix):
        """ 
        I think this just shows how much strongly each word is a part of each topic, 
        irrespective of the doc_term_matrix 
        """
        # # print some fitted terms
        # top_topic_terms = list(self.topic_model.top_topic_terms(self.vectorizer.id_to_term))
        # print("Some topics")
        # for i in range(5):
        #     print(i, ":\t", top_topic_terms[i][1][:3])        
        self.topic_model.termite_plot(doc_term_matrix, self.vectorizer.id_to_term, topics=-1)
        plt.title(title)
        plt.savefig(self.graphicspath+title+self.image_type)
        # plt.show()

    def print_topics(self):
        for topic_idx, top_terms in self.topic_model.top_topic_terms(
            self.vectorizer.id_to_term, topics=[0,1]):
        # print(topic_idx, top_terms)
            print("topic", topic_idx, ":", " ".join([str(i) for i in top_terms]))

    def clean_text(self, text):
        # https://towardsdatascience.com/topic-modeling-quora-questions-with-lda-nmf-aff8dce5e1dd
        '''Make text lowercase, remove text in square brackets, remove punctuation and remove words containing numbers.'''
        text = text.lower()
        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
        text = re.sub(r'\w*\d\w*', '', text)
        # TODO should we replace hyphons?
        # test = re.sub(r"\-", "")
        text = re.sub(r"\ +", ' ', text)
        return text

    def get_top_n_topics(self, df, doc_topic_matrix, top_n=3):
        # create topic name mapping
        topic_names = {}
        for idx, names in self.topic_model.top_topic_terms(self.vectorizer.id_to_term, topics=-1):
            topic_names[idx] = ("_".join(names[0:3]))

        all_topics = []
        for doc_idx, topics in self.topic_model.top_doc_topics(doc_topic_matrix, docs=-1, top_n=top_n):
            # print([:35], ":", topics)
            topics = (df.loc[doc_idx, "publication_date"],) + topics
            all_topics.append(topics)
        columns = ["publication_date"]

        # TODO take first 3 words as topic name
        # we should use them to replace the numbers!

        for i in range(len(all_topics[0])-1):
            columns += [f"topic_{i}"]
        topic_df = pd.DataFrame(all_topics, columns=columns)
        return topic_df, topic_names

    # # Just in case we want to remove certain types of token
    # def remove_tokens_on_match(self, doc):
    #     indexes = []
    #     for index, token in enumerate(doc):
    #         # TODO we don't really need to filter for PUNCT, bc we already do regex
    #         if (token.pos_  in ('PUNCT', "SPACE")):
    #             indexes.append(index)
    #     np_array = doc.to_array([LOWER, POS, ENT_TYPE, IS_ALPHA])
    #     np_array = numpy.delete(np_array, indexes, axis = 0)
    #     doc2 = Doc(doc.vocab, words=[t.text for i, t in enumerate(doc) if i not in indexes])
    #     doc2.from_array([LOWER, POS, ENT_TYPE, IS_ALPHA], np_array)
    #     return doc2


if __name__ == "__main__":
    # TODO design choice: use either textacy, or use sklearn implementation

    # TODO we can use spacy preprocessing for a count matrix
    # and run RfidfTransformer instead of Vectorizer?

    # TODO instead of our manual preprocessing, we might be able to just take some spacy property
    # a = ([tok.lemma_ for tok in spacy_doc] for spacy_doc in df.nlp)
    # corpus = textacy.Corpus(nlp, data=docs)

    # Following mostly this tutorial:
    # https://chartbeat-labs.github.io/textacy/build/html/api_reference/vsm_and_tm.html

    if not os.path.isdir("../experiments"):
        os.mkdir("../experiments")

    start_date=datetime.strptime("2020-03-25", "%Y-%m-%d")
    end_date=datetime.strptime("2020-04-06", "%Y-%m-%d")

    # Pass with representative fitting data
    # find and name topics
    print("Loading Data...\t", str(datetime.now()))
    representative_df = read_data.get_representative_df(
        n_samples=10,
        start_date=start_date,
        end_date=end_date
    )
    ta = TopicAnalyser()
    representative_df = ta.apply_nlp(representative_df)
    rep_doc_term_matrix = ta.get_doc_term_matrix(representative_df, fit=True)

    ## debug
    # TODO pring some stuff from rep_doc_term_matrix
    # for topic_idx, top_terms in ta.topic_model.top_topic_terms(
    #     ta.vectorizer.id_to_term, topics=[0,1]):
    #       print('topic', topic_idx, ':', '   '.join(top_terms))

    a = scipy.sparse.csr_matrix.toarray(rep_doc_term_matrix)
    ## debug

    rep_doc_topic_matrix = ta.get_doc_topic_matrix(rep_doc_term_matrix, fit=True)
    ta.visualize("Find Topics", rep_doc_term_matrix)

    # Pass with specific data set
    # TODO do some nicer prints
    # find topic distribution of a given day
    # convert to dataframe
    # save & run R script (maybe even via subprocess)
    print("Loading Data...\t", str(datetime.now()))
    df = read_data.get_body_df(
        start_date=start_date,
        end_date=end_date,
        articles_per_period=200,
        max_length=300
    )
    df = ta.apply_nlp(df)
    doc_term_matrix = ta.get_doc_term_matrix(df)
    doc_topic_matrix = ta.get_doc_topic_matrix(doc_term_matrix)

    all_topics, topic_names = ta.get_top_n_topics(df, doc_topic_matrix)

    # TODO make class function
    # TODO automatically name the topics!
    # transform topics
    all_topics = all_topics.loc[:, ["publication_date", "topic_0"]]
    df_gb = all_topics.groupby(by=["publication_date", "topic_0"])
    res = (pd.DataFrame(df_gb["topic_0"].
        apply(sum)).
        rename({"topic_0": "sum"}, axis=1).
        reset_index().
        rename({"topic_0": "main_topic"}, axis=1))
    # set nicer topic names
    res["main_topic"] = res["main_topic"].replace(topic_names)
    res.to_csv("src/topic_frequency.csv", index=False)

    # TODO ideally, call R from subprocess
    


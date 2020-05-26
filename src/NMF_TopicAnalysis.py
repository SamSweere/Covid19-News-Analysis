import sys
sys.path.append("src/")
import spacy
from sklearn.decomposition import NMF
import read_data
from datetime import datetime, timedelta
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
import unicodedata
import time
import shutil
np.random.seed(0)


# TODO should probably just inherit from model or something...
class TopicAnalyser:
    def __init__(self, n_topics):
        self.vectorizer = self.get_vectorizer()
        self.topic_model = self.get_nmf_model(n_topics)
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
        print("Cleaning Text")
        df["clean_body"] = df["body"].apply(lambda x: self.clean_text(x))
        # We rely on spacy for all preprocessing
        # TODO not sure if we are feeding textacy exactly what it wants
        print("Apply NLP")
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

    def get_nmf_model(self, n_topics):
        # TODO how can we get this to converge?
        model = textacy.tm.TopicModel("nmf", n_topics=n_topics, max_iter=10000)
        return model

    def get_custom_stopwords(self):
        """ remove news outlet names and the like """
        custom = ["say", "news", "reuters", "cbcca", "getty", "reuter", "get", "am", "pm", "et", "go", "like"]
        custom_stopwords = stopwords.words('english') + ["-PRON-"] + custom
        return custom_stopwords

    def get_doc_term_matrix(self, df, fit=False):
        """ 
        Get doc_term_matrix for docs in df. feed docs as nested list; maintains order so we can just keep using our df
        """
        # print(f"Weighting formula: {self.vectorizer.weighting}")  # tfidf
        # TODO get all the lemmas instead of the text here!
        # for i in df["nlp"]:
        #     sent = []
        #     for j in i:
        #         sent.append(j.lemma_)
        #     docs.append

        # docs = [j.lemma for i in df["nlp"] for j in i]
        # docs = [i.doc.text for i in df["nlp"]]
        print("Getting Doc-Term Matrix...\t", str(datetime.now()))
        custom_stopwords = self.get_custom_stopwords()
        my_terms_list=[[tok.lemma_ for tok in doc if tok.lemma_ not in custom_stopwords] 
            for doc in df["nlp"]]
        if fit:
            self.vectorizer.fit(my_terms_list)
        doc_term_matrix = self.vectorizer.transform(my_terms_list)
        # print("Some Terms:")
        # print(self.vectorizer.terms_list[:10])

        return doc_term_matrix

    def get_doc_topic_matrix(self, doc_term_matrix, fit=False):
        print("Getting Doc-Topic Matrix...\t", str(datetime.now()))
        if fit:
            self.topic_model.fit(doc_term_matrix)
        doc_topic_matrix = self.topic_model.transform(doc_term_matrix)
        print(f"Doc-Topic matrix shape: {doc_topic_matrix.shape}")
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
        plt.tight_layout()
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
        # text = unicodedata.normalize("NFKD", text)
        text = re.sub(r'\[.*?\]', "", text)
        text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
        text = re.sub(r'\w*\d\w*', "", text)
        text = re.sub(r'\n|\t', "", text)
        text = re.sub('[^a-zA-Z0-9 \n\.]', "", text)
        # TODO should we replace hyphons?
        # test = re.sub(r"\-", "")
        # TODO write a better regex for this stuff
        text = text.replace("\xa0", "")
        text = text.replace("’s", "")
        text = text.replace("’", "")
        text = text.replace('"', "")
        text = text.replace("”", "")
        text = text.replace("“", "")
        # text = text.replace("'s", "")
        # text = text.replace("'", "")
        text = re.sub(r"\ +", ' ', text)
        return text

    def get_top_n_topics(self, df, doc_topic_matrix, top_n=3):
        # create topic name mapping
        print("Getting Top N topics...\t", str(datetime.now()))
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

    def get_topics_per_day(self, all_topics, topic_names):
        all_topics = all_topics.loc[:, ["publication_date", "topic_0"]]
        df_gb = all_topics.groupby(by=["publication_date", "topic_0"])
        res = (pd.DataFrame(df_gb["topic_0"].
            agg(["sum"])).
            reset_index().
            rename({"topic_0": "main_topic"}, axis=1))

        # this is assuming we do day by day
        total_day = res.groupby(by=["publication_date"])["sum"].sum()[0]
        res["mean"] = res["sum"]/total_day
        # set nicer topic names
        res["main_topic"] = res["main_topic"].replace(topic_names)
        # "+str(datetime.now())+"
        res.to_csv("src/TopicAnalysis/topic_frequency.csv", index=False)
        return res

def run_and_save(start_date, end_date, articles_per_period, max_length):
    c_date = start_date

    # Name 
    # run_name = "s_" + start_date.strftime("%d_%m_%Y") + "_e_" \
    #     + end_date.strftime("%d_%m_%Y") + "_app_" + str(articles_per_period) \
    #     + "_ml_" + str(max_length) + "_" + datetime.now().strftime("d_%d_%m_t_%H_%M")
    # run_name = datetime.now().strftime("d_%d_%m_t_%H_%M")
    run_name = "ta_run_" + datetime.now().strftime("d_%d_%m_t_%H_%M")
    print(f"RUN NAME: {run_name}")

    folder_path = "data/0TopicAnalysis/"+run_name
    if os.path.isdir(folder_path):
        shutil.rmtree(folder_path)
    os.mkdir(folder_path)

    print("Start run from date: " + start_date.strftime("%d_%m_%Y") + " to date: " + end_date.strftime("%d_%m_%Y"))
    print("Articles per period: " + str(articles_per_period))
    print("Max Length per article: " + str(max_length))

    start_time = time.process_time()

    # Increase until we hit the last day
    while c_date < end_date:
        print("Running day:",c_date.strftime("%d_%m_%Y"))
        print("Loading Data...\t", str(datetime.now()))
        df = read_data.get_body_df(
            start_date=c_date,
            end_date=c_date,
            articles_per_period=articles_per_period, #700,
            max_length=max_length
        )
        if df.shape[0] == 0:
            c_date += timedelta(days=1)
            continue

        df = ta.apply_nlp(df)
        doc_term_matrix = ta.get_doc_term_matrix(df)
        doc_topic_matrix = ta.get_doc_topic_matrix(doc_term_matrix)
        all_topics, topic_names = ta.get_top_n_topics(df, doc_topic_matrix)
        ta.visualize("Find Topics", doc_term_matrix)
        topics_per_day = ta.get_topics_per_day(all_topics, topic_names)
        # print(topics_per_day)

        file_name = c_date.strftime("%d_%m_%Y")
        topics_per_day.to_csv(folder_path + "/" + file_name +".csv", index=False)

            # Increase the day
        c_date += timedelta(days=1)
    
    print("Multiple day run finished")
    elapsed_time = time.process_time() - start_time
    print("Time elapsed: " + str(round(elapsed_time,2)) + " seconds")
    pass


if __name__ == "__main__":
    # TODO instead of our manual preprocessing, we might be able to just take some spacy property
    # a = ([tok.lemma_ for tok in spacy_doc] for spacy_doc in df.nlp)
    # corpus = textacy.Corpus(nlp, data=docs)

    # Following mostly this tutorial:
    # https://chartbeat-labs.github.io/textacy/build/html/api_reference/vsm_and_tm.html

    if not os.path.isdir("../experiments"):
        os.mkdir("../experiments")

    ta = TopicAnalyser(n_topics=10)
    start_date = datetime.strptime("2020-02-01", "%Y-%m-%d")
    # start_date = datetime.strptime("2020-04-01", "%Y-%m-%d")
    end_date = datetime.strptime("2020-04-06", "%Y-%m-%d")

    # # Pass with representative fitting data to find and name topics
    print("Loading Data...\t", str(datetime.now()))
    representative_df = read_data.get_representative_df(
        n_samples=20000,
        start_date=start_date,
        end_date=end_date,
        max_length=2000
    )
    representative_df = ta.apply_nlp(representative_df)
    rep_doc_term_matrix = ta.get_doc_term_matrix(representative_df, fit=True)
    rep_doc_topic_matrix = ta.get_doc_topic_matrix(rep_doc_term_matrix, fit=True)
    ta.visualize("Find Topics", rep_doc_term_matrix)
    print("Found our topics...\t", str(datetime.now()))

    # Run day by day
    run_and_save(start_date, end_date, articles_per_period=1000,
        max_length=1000)
        
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
from nltk.corpus import stopwords
import matplotlib.pyplot as plt


# TODO should probably just inherit from model or something...
class TopicAnalyser:
    def __init__(self):
        self.vectorizer = self.get_vectorizer()
        self.model = self.get_nmf_model()

    def analyze(self):
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
        model = textacy.tm.TopicModel("nmf", n_topics=20)
        return model

    def get_doc_term_matrix(self, df):
        print(f"Weighting formula: {self.vectorizer.weighting}")  # tfidf
        # doc_term_matrix = vectorizer.fit_transform(corpus)
        docs = [i.doc.text for i in df["nlp"]]
        my_terms_list=[[tok  for tok in doc.split() if tok not in stopwords.words('english') ] for doc in docs]
        doc_term_matrix = self.vectorizer.fit_transform(my_terms_list)
        print("Some Terms:")
        print(self.vectorizer.terms_list[:10])

        return doc_term_matrix

    def get_doc_topic_matrix(self, doc_term_matrix):
        # TODO when should we fit this, when should we transform it?
        self.model.fit(doc_term_matrix)
        doc_topic_matrix = self.model.transform(doc_term_matrix)
        print(f"Doc topic matrix shape: {doc_topic_matrix.shape}")
        return doc_topic_matrix

    def visualize(self):
        top_topic_terms = list(self.model.top_topic_terms(self.vectorizer.id_to_term))
        print(top_topic_terms)
        self.model.termite_plot(doc_term_matrix, self.vectorizer.id_to_term, topics=-1)
        plt.show()
        # TODO we should do this plydavis stuff here as well, it's pretty cool

    def print_topics(self):
        for topic_idx, top_terms in self.model.top_topic_terms(
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
        text = re.sub(r"\ +", ' ', text)
        return text

    # Just in case we want to remove certain types of token
    def remove_tokens_on_match(self, doc):
        indexes = []
        for index, token in enumerate(doc):
            # TODO we don't really need to filter for PUNCT, bc we already do regex
            if (token.pos_  in ('PUNCT', "SPACE")):
                indexes.append(index)
        np_array = doc.to_array([LOWER, POS, ENT_TYPE, IS_ALPHA])
        np_array = numpy.delete(np_array, indexes, axis = 0)
        doc2 = Doc(doc.vocab, words=[t.text for i, t in enumerate(doc) if i not in indexes])
        doc2.from_array([LOWER, POS, ENT_TYPE, IS_ALPHA], np_array)
        return doc2


if __name__ == "__main__":
    # TODO design choice: use either textacy, or use sklearn implementation

    # TODO we can use spacy preprocessing for a count matrix
    # and run RfidfTransformer instead of Vectorizer?

    # TODO instead of our manual preprocessing, we might be able to just take some spacy property
    # a = ([tok.lemma_ for tok in spacy_doc] for spacy_doc in df.nlp)
    # corpus = textacy.Corpus(nlp, data=docs)

    # Following mostly this tutorial:
    # https://chartbeat-labs.github.io/textacy/build/html/api_reference/vsm_and_tm.html

    if not os.path.isdir("experiments"):
        os.mkdir("experiments")

    print("Loading Data...\t", str(datetime.now()))
    start_date=datetime.strptime("2020-04-03", "%Y-%m-%d")
    end_date=datetime.strptime("2020-04-06", "%Y-%m-%d")
    df = read_data.get_body_df(
        start_date=start_date,
        end_date=end_date,
        articles_per_period=2,
        max_length=300
    )

    ta = TopicAnalyser()
    df = ta.apply_nlp(df)
    doc_term_matrix = ta.get_doc_term_matrix(df)
    doc_topic_matrix = ta.get_doc_topic_matrix(doc_term_matrix)
    ta.visualize()

    # TODO
    # find automatic topic names for any given day
    # find topic distribution of a given day
    # convert to dataframe
    # save & run R script (maybe even via subprocess)






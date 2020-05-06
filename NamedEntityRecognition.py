""" Named Entity Recognition """

import read_data
from datetime import datetime, timedelta
import spacy  # the spacy nets have been trained on OntoNotes 5.0
from spacy import displacy
from collections import Counter
import pandas as pd
from itertools import chain
import visualization
import neuralcoref
from datetime import datetime


class NamedEntityRecognizer:
    def __init__(self):
        pass

    def spacy_preprocessing(self, df, nlp):
        print("Starting NLP...", str(datetime.now()))
        df["nlp"] = df["body"].apply(nlp)
        print("Starting NLP on resolved corpora", str(datetime.now()))
        df["nlp_resolved"] = df["nlp"].apply(lambda x: nlp(x._.coref_resolved))
        return df

    def most_common_entities(self, df, visualize):
        """ 
        @param df: DataFrame with "nlp" column containing spacy preprocessing
        @param visualization: create bar chart race
        """
        df_most_common = self.find_most_common_entities(df, "nlp_resolved", n_common=1, entity_type="PERSON")
        df_most_common["publication_date"] = df["publication_date"]
        df_most_common = df_most_common.groupby(by=["publication_date", "most_common_1"])
        df_most_common = df_most_common.agg(sum).sort_values(by="most_common_1_num", ascending=False).reset_index()
        df_most_common = df_most_common.groupby(by=["publication_date"])
        df_most_common = df_most_common.head(10)
        if visualize:
            print("Starting Visualization...")
            visualization.animate_NER(df_most_common)
        return(df_most_common)


    def find_most_common_entities(self, df, nlp_doc_colname:str, n_common:int, entity_type:str):
        """ 
        get most common entities for a series of articles on a certain date 
        @param df: data frame containing our articles
        @param nlp_doc_colname: name of column containing nlp-processed documents
        @param n_common: number of most frequent to keep
        """
        # TODO find most common entities on resolved text!!
        most_common = []
        for d in df[nlp_doc_colname]:

            # get all entities
            item_labels = [(x.text, x.label_) for x in d.ents]

            # select only entities that match our type
            items = [i for (i, l) in item_labels if l==entity_type]
            most_common += Counter(items).most_common(n_common)
        colnames = []
        for i in range(1, n_common+1):
            colnames.append(f"most_common_{i}")
            colnames.append(f"most_common_{i}_num")
        df_most_common = pd.DataFrame.from_records(most_common, columns=colnames)
        return df_most_common
    
    def count_most_frequent(self, group):
        # current approach: most common of the most common - should be ok, I think
        items = [[i[0]] * i[1] for i in group if not (i is None)]
        items = list(chain.from_iterable(items))
        most_common = Counter(items).most_common(1)
        return most_common[0][0]  # get only the phrase for now

    def show_problems(self, article, visualize=False):
        """ 
        show some of our current problems in sentiment analyses 
        @param article: result of spacy nlp applied to the body of an article
        """
        # we need to further resolve entities
        item_labels = [(x.text, x.label_) for x in article.ents]
        print(item_labels)

        if visualize:
            # dependency parsing
            displacy.serve(article, style='dep', options = {'distance': 120})

            # entity recognition
            displacy.serve(article, style='ent')


if __name__ == "__main__":


    print("Loading Data...")
    df = read_data.get_body_df(
        start_date=datetime.strptime("2020-03-15", "%Y-%m-%d"),
        end_date=datetime.strptime("2020-04-06", "%Y-%m-%d"),
        articles_per_period=300,
    )

    nlp = spacy.load("en_core_web_sm")  # "eng_core_web_lg" for better but slower results
    coref = neuralcoref.NeuralCoref(nlp.vocab, greedyness=0.5)
    nlp.add_pipe(coref, name='neuralcoref')
    
    # df["nlp"][0]
    # x = coref(article)
    # x._.coref_resolved
    # TODO this is how we should be doing things
    # def coref_res_component(doc):
    #     doc = coref(doc)
    #     return doc._.corefresolved_

    merge_ents = nlp.create_pipe("merge_entities")
    nlp.add_pipe(merge_ents)

    # TODO how do we get a proper custom pipeline??
    # custom pipeline
    nlp.pipe_names
    
    NER = NamedEntityRecognizer()
    df = NER.spacy_preprocessing(df, nlp)
    # might be a lot faster if we merge all articles of a day into one document?
    df_most_common = NER.most_common_entities(df, visualize=True)

    # article = df["nlp"][0]
    # NER.show_problems(article)

    
# TODO a lot of confusion for company names etc.
# TODO Coreference Resolution
# TODO Entity Resolution

# TODO example: doesn't understand that Johnson is same as Boris Johnson
# TODO for word vectors & similarity, we might have to train new model to catch stuff like "Coronavirus"
# TODO basically: tailor model to our data set


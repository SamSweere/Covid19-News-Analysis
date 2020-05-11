""" Named Entity Recognition """

import read_data
from datetime import datetime, timedelta
import spacy  # the spacy nets have been trained on OntoNotes 5.0
from spacy import displacy
from collections import Counter
import pandas as pd
from itertools import chain
import visualization
import bar_chart_race
import neuralcoref
from datetime import datetime


# currently we can process about 500 articles per minute


class NamedEntityRecognizer:
    def __init__(self):
        pass

    def spacy_preprocessing(self, df, nlp):
        print("Starting NLP...", str(datetime.now()))
        df["nlp"] = df["body"].apply(nlp)
        df["nlp_resolved"] = df["nlp"].apply(lambda x: nlp(x._.coref_resolved))
        print("Completed NLP by...", str(datetime.now()))
        return df

    def most_common_entities(self, df, visualize):
        """ 
        @param df: DataFrame with "nlp" column containing spacy preprocessing
        @param visualization: create bar chart race
        """

        # add entities for each publication date
        df_most_common = df.groupby(by=["publication_date", "most_common_1"])
        df_most_common = df_most_common.agg(sum).sort_values(by="most_common_1_num", ascending=False).reset_index()

        # select top 10 for each publication date
        df_most_common = df_most_common.groupby(by=["publication_date"])
        df_most_common = df_most_common.head(10)

        if visualize:
            print("Starting Visualization...")
            visualization.animate_NER(df_most_common)
            bar_chart_race.create_barchart_race(df_most_common)
        
        return df_most_common


    def find_most_common_entities(self, df, nlp_doc_colname:str, entity_type="PERSON"):
        """ Find most common entity for each article """
        def find_most_common_entity(article):
            items = [x.text for x in article.ents if x.label_ == entity_type]
            if len(items)==0:
                return ()
            else:
                return Counter(items).most_common(1)[0]
        df[["most_common_1", "most_common_1_num"]] = pd.DataFrame.from_records(df[nlp_doc_colname].apply(find_most_common_entity))
        return df


    # def find_most_common_entities(self, df, nlp_doc_colname:str, n_common:int, entity_type:str):
    #     """ 
    #     get most common entities for a series of articles on a certain date 
    #     @param df: data frame containing our articles
    #     @param nlp_doc_colname: name of column containing nlp-processed documents
    #     @param n_common: number of most frequent to keep # TODO for now only one seems to work!
    #     """
    #     # TODO should we normalize by total number of words?

    #     # TODO what exactly do we want to measure here?
    #     # TODO try different approach

    #     most_common = []
    #     # append n most common for each document
    #     for d in df[nlp_doc_colname]:
    #         # get entities of our type
    #         items = [x.text for x in d.ents if x.label_ == entity_type]
    #         # n most common items
    #         most_common += Counter(items).most_common(n_common)
    #     colnames = []
    #     for i in range(1, n_common+1):
    #         colnames.append(f"most_common_{i}")
    #         colnames.append(f"most_common_{i}_num")
    #     df_most_common = pd.DataFrame.from_records(most_common, columns=colnames)
    #     return df_most_common
    
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
    
    def load_preloaded(self, path):
        return pd.read_csv(path)


if __name__ == "__main__":


    print("Loading Data...")
    df = read_data.get_body_df(
        start_date=datetime.strptime("2020-04-04", "%Y-%m-%d"),
        end_date=datetime.strptime("2020-04-06", "%Y-%m-%d"),
        articles_per_period=1000,
    )

    # only load what we need for ner
    nlp = spacy.load("en_core_web_sm", disable=["parser", "tagger"])  # "eng_core_web_lg" for better but slower results
    nlp.add_pipe(nlp.create_pipe("sentencizer"))
    # TODO make use of merger
    # merge_ents = nlp.create_pipe("merge_entities")
    # nlp.add_pipe(merge_ents)
    coref = neuralcoref.NeuralCoref(nlp.vocab, greedyness=0.5)
    nlp.add_pipe(coref, name='neuralcoref')

    # custom pipeline
    print("Pipeline: --------------")
    print(nlp.pipe_names)
    print("------------------------")
    
    NER = NamedEntityRecognizer()
    
    # might be a lot faster if we merge all articles of a day into one document?

    # TODO look into spacy pipe pattern
    # docs = df["body"]
    # gen = nlp.pipe(docs, 20, 5)
    # a = [i for i in gen]

    # df = NER.load_preloaded()
    df_pp = NER.spacy_preprocessing(df, nlp)
    # df_pp.to_csv("df_pp.csv")

    # TODO run later
    # df_test = NER.find_most_common_entities(df_pp, "nlp_resolved", entity_type="PERSON")
    # df_test = df_test[["publication_date", "most_common_1", "most_common_1_num"]]
    # df_most_common = NER.most_common_entities(df_test, visualize=True)

    # article = df["nlp"][0]
    # sentence = "On March 26, Johnson revealed he had tested positive and that he had been dealing with symptoms since that date."
    # displacy.serve(nlp(sentence), style='dep', options = {'distance': 120})
    # NER.show_problems(article)

    
# TODO a lot of confusion for company names etc.
# TODO Entity Resolution

# TODO example: doesn't understand that Johnson is same as Boris Johnson
# TODO for word vectors & similarity, we might have to train new model to catch stuff like "Coronavirus"
# TODO basically: tailor model to our data set


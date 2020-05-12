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
import spacy_dbpedia_spotlight
import os


# currently we can process about 500 articles per minute


class NamedEntityRecognizer:
    def __init__(self):
        pass

    def spacy_preprocessing(self, df, model_size):
        # TODO build separate pipelines for dbpedia vs. standard NER
        # build param for _sm / _lg model
        assert model_size in ["sm", "lg"]
        nlp = spacy.load(f"en_core_web_{model_size}")  # "eng_core_web_lg" for better but slower results
        # TODO do we need/want entity merger?
        # merge_ents = nlp.create_pipe("merge_entities'")
        # nlp.add_pipe(merge_ents)
        coref = neuralcoref.NeuralCoref(nlp.vocab, greedyness=0.5)
        nlp.add_pipe(coref, name='neuralcoref')
        print("Starting NLP...\t", str(datetime.now()))
        print("------------------------")
        print(nlp.pipe_names)
        print("------------------------")
        # df["nlp"] = [i for i in nlp.pipe(df["body"], batch_size=10, n_threads=6)]  # didn't really speed up anything
        df["nlp"] = df["body"].apply(nlp)
        return df

    def spacy_ner(self, df, model_size):
        """ Use spacy CNN NER """
        # TODO should we try some of https://github.com/explosion/spaCy/tree/2d249a9502bfc5d3d2111165672d964b1eebe35e/bin/wiki_entity_linking
        assert model_size in ["sm", "lg"]
        nlp_pp = spacy.load(f"en_core_web_{model_size}")
        print("Starting NLP Coref\t", str(datetime.now()))
        print("------------------------")   
        print("Spacy NER pipeline:", nlp_pp.pipe_names)   
        print("------------------------")   
        df["nlp_resolved"] = df["nlp"].apply(lambda x: nlp_pp(x._.coref_resolved))
        print("Completed NLP by...\t", str(datetime.now()))
        return df

    def dbpedia_ner(self, df, model_size):
        """ Use DBPedia matching """
        assert model_size in ["sm", "lg"]
        nlp_pp = spacy.load(f"en_core_web_{model_size}", disable=["ner"])
        # TODO we might be throwing out a lot of stuff in entity_linker (spacy_dbpedia_spotlight), maybe go check?
        spacy_dbpedia_spotlight.load('en', nlp_pp)
        print("Starting NLP Coref", str(datetime.now()))
        print("------------------------")   
        print("DBPedia NER pipeline:", nlp_pp.pipe_names)   
        print("------------------------")
        def inner(x):
            return nlp_pp(x)
        df["nlp_resolved"] = df["nlp"].apply(lambda x: inner(x._.coref_resolved))
        print("Completed NLP by...\t", str(datetime.now()))
        return df

    def most_common_entities(self, df):
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
        return df_most_common

    def visualize(self, df_most_common, start_date, end_date):
        print("Starting Visualization...\t", str(datetime.now()))
        visualization.animate_NER(df_most_common)
        bar_chart_race.create_barchart_race(df_most_common, start_date, end_date)
                
    def find_most_common_entities(self, df, nlp_doc_colname:str, entity_type:str):
        """ Find most common entity for each article """
        def find_most_common_entity(article):
            if article is None:
                return tuple()
            all_items = [(x.text, x.label_) for x in article.ents]
            items = [x.text for x in article.ents if entity_type in x.label_]
            most_common = Counter(items).most_common(1)
            if len(most_common) == 0:
                return tuple()
            else:
                return most_common[0]
        # For some reason all the NLP resolved are None!!
        df[["most_common_1", "most_common_1_num"]] = pd.DataFrame.from_records(
            df[nlp_doc_colname].apply(find_most_common_entity))
        return df

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

        # run as:
        # article = df["nlp"][0]
        # sentence = "On March 26, Johnson revealed he had tested positive and that he had been dealing with symptoms since that date."
        # displacy.serve(nlp(sentence), style='dep', options = {'distance': 120})
        # NER.show_problems(article)

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

    def cum_sum_df(self, df):
        """ return df with monotonically increasing entity mentions """
        # TODO this does not ensure continuity in case some entity is missing entirely on some days
        # should be kinda rare but still possible...
        df = df.sort_values(by=["publication_date", "most_common_1"])
        df_gb = df.groupby(by=["most_common_1"])
        df["cum_sum"] = df_gb.most_common_1_num.transform(pd.Series.cumsum)
        return df


if __name__ == "__main__":

    if not os.path.isdir("experiments"):
        os.mkdir("experiments")

    print("Loading Data...\t", str(datetime.now()))
    start_date=datetime.strptime("2020-03-01", "%Y-%m-%d")
    end_date=datetime.strptime("2020-04-06", "%Y-%m-%d")
    df = read_data.get_body_df(
        start_date=start_date,
        end_date=end_date,
        articles_per_period=300,
        max_length=300
    )

    NER = NamedEntityRecognizer()
    # might be a lot faster if we merge all articles of a day into one document?
    # df = NER.load_preloaded()

    # TODO we can't keep saving all the preliminary stages in our data frame, we'll run out of ram
    df_pp = NER.spacy_preprocessing(df, model_size="sm")
    df_pp = NER.dbpedia_ner(df_pp, model_size="sm")
    df_pp = NER.find_most_common_entities(df_pp, "nlp_resolved", entity_type="Person")  # entity "OfficeHolder" is quite nice, "Person" works as well
    df_pp = df_pp[["publication_date", "most_common_1", "most_common_1_num"]]
    df_most_common = NER.most_common_entities(df_pp)
    df_most_common = NER.cum_sum_df(df_most_common)
    df_most_common.to_csv("df_most_common"+str(datetime.now())+".csv")
    print(df_most_common)
    NER.visualize(df_most_common, start_date, end_date)


# TODO a lot of confusion for company names etc.
# TODO for word vectors & similarity, we might have to train new model to catch stuff like "Coronavirus"
# TODO basically: tailor model to our data set


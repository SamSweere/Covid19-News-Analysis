""" Named Entity Recognition """

import sys
import read_data
from datetime import datetime, timedelta
import spacy  # the spacy nets have been trained on OntoNotes 5.0
from spacy import displacy
from collections import Counter
import pandas as pd
from itertools import chain
import neuralcoref
from datetime import datetime
# import spacy_dbpedia_spotlight
import numpy as np
from collections import namedtuple
import os

sys.path.append("src/visualization/")
import matplotlib_viz as viz
import bar_chart_race

# custom adaption of spacy_dbpedia_spotlight
sys.path.append("src/spacy_dbpedia_spotlight/")
import entity_linker
import initialize

# TODO I think we get an error bc we don't have the right dbpedia spotlight version?
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
        coref = neuralcoref.NeuralCoref(nlp.vocab, greedyness=0.4)
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
        initialize.load('en', nlp_pp)
        print("Starting NLP Coref", str(datetime.now()))
        print("------------------------")   
        print("DBPedia NER pipeline:", nlp_pp.pipe_names)   
        print("------------------------")
        def inner(x):
            return nlp_pp(x)
        df["nlp_resolved"] = df["nlp"].apply(lambda x: inner(x._.coref_resolved))
        print("Completed NLP by...\t", str(datetime.now()))
        return df

    def sum_period_most_common_entities(self, df):
        """ 
        @param df: DataFrame with "nlp" column containing spacy preprocessing
        @param visualization: create bar chart race
        """
        # sum up entities for each publication date
        df_gb = df.groupby(by=["publication_date", "most_common_1"])
        df_most_common = df_gb.agg(sum).reset_index()
        return df_most_common

    def select_most_common_per_period(self, df_most_common):
        # select top 10 for each publication date by cum_sum
        df_most_common = df_most_common.sort_values(by=["publication_date", "cum_sum"], ascending=False).reset_index()
        df_most_common = df_most_common.groupby(by=["publication_date"])
        df_most_common = df_most_common.head(10)
        return df_most_common

    def visualize(self, df_most_common, start_date, end_date):
        print("Starting Visualization...\t", str(datetime.now()))
        viz.animate_NER(df_most_common)
        bar_chart_race.create_barchart_race(df_most_common, start_date, end_date)
                
    def find_most_common_entities(self, df, nlp_doc_colname:str, entity_type:str):
        """ Find most common entity for each article """
        # TODO put a similarity threshold in here somewhere!
        def find_most_common_entity(article_raw):
            article = article_raw[nlp_doc_colname]
            if article is None:
                return tuple()
            all_items = []
            items = []
            for x in article.ents:
                if x.text.isspace():
                    continue
                text = x.text
                url = x.label_.split(" ")[0]
                dbpedia_labels = x.label_.split(" ")[1]
                # TODO maybe use namedtuple here
                all_items.append((text, url, dbpedia_labels))
                
                # TODO perhaps look for a more general solution here?
                # Idea: find a good compromise between standard NER and DBPedia
                # Washington case: Trust standard NER for washington GPE
                if text.lower() == "washington":
                    standard_ner_ents = {i.text.lower(): i.label_ for i in article_raw["nlp"].ents}
                    if "washington" in [i.lower() for i in standard_ner_ents.keys()]:
                        if standard_ner_ents["washington"] == "GPE":
                            dbpedia_labels = "Location_State"

                if entity_type in dbpedia_labels:
                    # append last bit of url to make sure we don't get duplicates
                    last_url_tag = url.rfind("/")
                    entity_name = url[last_url_tag+1:].replace("_", " ")
                    items.append(entity_name)

            # items = [x.text for x in article.ents if entity_type in x.label_]
            most_common = Counter(items).most_common(1)
            if len(most_common) == 0:
                return tuple()
            else:
                return most_common[0]
        
        # For some reason all the NLP resolved are None!!
        df[["most_common_1", "most_common_1_num"]] = pd.DataFrame.from_records(
            df.apply(lambda x: find_most_common_entity(x), axis=1))  # apply to each row
        df.dropna(inplace=True)     
        return df

    def fill_entity_gaps(self, df_most_common):
        """
        make sure we keep an absent but known entity around
        to avoid the bar from disappearing in bar chart race 
        """
        assert set(df_most_common.columns) == set(['publication_date', 'most_common_1', 'most_common_1_num'])

        # identify missing values
        all_entities = set(df_most_common["most_common_1"].unique())
        df_gb = df_most_common.groupby(by=["publication_date"])
        date_entities = df_gb["most_common_1"].aggregate(lambda x: set(x))
        
        # collect rows to add
        rows_to_add = []
        for i, d_e in enumerate(date_entities):
            missing_ents = all_entities.difference(d_e)
            date = date_entities.index[i]
            for e in missing_ents:
                row = (date, e, 0.0)
                rows_to_add.append(row)

        # add missing rows and sort again
        df_missing = pd.DataFrame(rows_to_add, columns=df_most_common.columns)
        df_most_common = pd.concat((df_most_common, df_missing), axis=0).reset_index()
        df_most_common.sort_values(by=["publication_date", "most_common_1"], inplace=True)

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

        # run as:
        # article = df["nlp"][0]
        # sentence = "On March 26, Johnson revealed he had tested positive and that he had been dealing with symptoms since that date."
        # displacy.serve(nlp(sentence), style='dep', options = {'distance': 120})
        # NER.show_problems(article)

        # we need to further resolve entities
        item_labels = [(x.text, x.label_) for x in article.ents if not x.text.isspace()]
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
        df["cum_sum"] = df_gb["most_common_1_num"].apply(pd.Series.cumsum)
        df.sort_values(by=["publication_date", "cum_sum"], ascending=False, inplace=True)
        return df


if __name__ == "__main__":

    if not os.path.isdir("src/experiments"):
        os.mkdir("src/experiments")

    if not os.path.isdir("src/logs"):
        os.mkdir("src/logs")

    print("Loading Data...\t", str(datetime.now()))
    start_date=datetime.strptime("2020-04-01", "%Y-%m-%d")
    end_date=datetime.strptime("2020-04-06", "%Y-%m-%d")
    df = read_data.get_body_df(
        start_date=start_date,
        end_date=end_date,
        articles_per_period=30, #700,
        max_length=300
    )

    NER = NamedEntityRecognizer()
    # might be a lot faster if we merge all articles of a day into one document?
    # df = NER.load_preloaded()

    # TODO log each visualization in  anew folder and save specification

    # TODO we can't keep saving all the preliminary stages in our data frame, we'll run out of ram
    # TODO: changed the model size to sm
    df_pp = NER.spacy_preprocessing(df, model_size="sm") # model_size="lg")
    df_pp = NER.dbpedia_ner(df_pp,model_size="sm") #model_size="lg")
    
    # TODO make this a debugging function
    # TODO - Misclassification debugging
    # Check how come we get 'Washington' as a person this much!
    # Functionality for checking a given entity
    debug = False
    if debug:
        for i in range(df_pp.shape[0]):
            if ("Washington" in df_pp.loc[i, :].nlp.text):
                print(i)

        # show entities of a given article
        import requests
        example = df_pp.loc[16, :]
        print(example.nlp.text)
        # standard NER
        items = [(i.text, i.label_) for i in example.nlp.ents]
        print(items)
        print()
        # dbpedia NER
        dbpedia_items = [(i.text, i.label_, i.start, i.end) for i in example.nlp_resolved.ents]
        print(dbpedia_items)

        for i in dbpedia_items:
            print(i)
        # query dbpedia with example text

        # check coref greedyness
        nlp = spacy.load(f"en_core_web_sm")  # "eng_core_web_lg" for better but slower results
        coref = neuralcoref.NeuralCoref(nlp.vocab, greedyness=0.4)
        nlp.add_pipe(coref, name='neuralcoref')
        res = nlp(example.nlp.text)
        ungreedy_text = res._.coref_resolved
        print(ungreedy_text)

        # TODO the neuralcoref entities look better than the NER ones, can we use those?
        base_url = "http://localhost:2222/rest"
        response = requests.get(f"{base_url}/annotate",
            headers={'accept': 'application/json'},
            params={'text': "nursing home residents in one Washington state facility"})
        data = response.json()
        print(data)
        for i in data.get("Resources"):
            print(str(i)+"\n")
        # look at result closely to make sure we pick right entity

    df_pp = NER.find_most_common_entities(df_pp, "nlp_resolved", entity_type="Person")  # entity "OfficeHolder" is quite nice, "Person" works as well
    df_pp = df_pp[["publication_date", "most_common_1", "most_common_1_num"]]
    df_most_common = NER.sum_period_most_common_entities(df_pp)
    df_most_common = NER.fill_entity_gaps(df_most_common)
    df_most_common = NER.cum_sum_df(df_most_common)
    df_most_common = NER.select_most_common_per_period(df_most_common)

    # TODO we need to do everything in the right order!

    df_most_common.to_csv("src/logs/df_most_common"+str(datetime.now())+".csv")
    print(df_most_common)
    NER.visualize(df_most_common, start_date, end_date)

    pass
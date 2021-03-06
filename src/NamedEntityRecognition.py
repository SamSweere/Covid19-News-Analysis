""" Named Entity Recognition """

import sys
from datetime import datetime, timedelta
import spacy  # the spacy nets have been trained on OntoNotes 5.0spacy.prefer_gpu()
# spacy.prefer_gpu()
from spacy import displacy
from collections import Counter
import pandas as pd
from itertools import chain
import neuralcoref
from datetime import datetime
import time
# import spacy_dbpedia_spotlight
import numpy as np
from collections import namedtuple
from copy import deepcopy
import os
import requests

sys.path.append("src/")
import read_data

sys.path.append("src/visualization/")
import visualization.get_viz_data as get_viz_data
import visualization.matplotlib_viz as viz
import visualization.bar_chart_race as bar_chart_race

# custom adaption of spacy_dbpedia_spotlight
sys.path.append("src/spacy_dbpedia_spotlight/")
from spacy_dbpedia_spotlight import entity_linker 
from spacy_dbpedia_spotlight import initialize

from sentiment import target_based_sentiment
from sentiment import general_sentiment


class NamedEntityRecognizer:
    def __init__(self, model_size, with_sentiment=False):
        assert model_size in ["sm", "lg"]
        self.nlp = spacy.load(f"en_core_web_{model_size}")  # "eng_core_web_lg" for better but slower results
        coref = neuralcoref.NeuralCoref(self.nlp.vocab, greedyness=0.4)
        self.nlp.add_pipe(coref, name='neuralcoref')


        self.nlp_pp = spacy.load(f"en_core_web_{model_size}")

        self.nlp_dbp = spacy.load(f"en_core_web_{model_size}", disable=["ner"])
        initialize.load('en', self.nlp_dbp)


        self.nlp_nr = spacy.load(f"en_core_web_{model_size}")

        if with_sentiment:
            # Load target based sentiment
            self.tsa = target_based_sentiment.TargetSentimentAnalyzer()  

            # Load general based sentiment
            self.gsa = general_sentiment.GeneralSentimentAnalyzer()

    def spacy_preprocessing(self, df):
        # TODO build separate pipelines for dbpedia vs. standard NER
        # build param for _sm / _lg model
        
        # TODO do we need/want entity merger?
        # merge_ents = nlp.create_pipe("merge_entities'")
        # nlp.add_pipe(merge_ents)
        print("Starting NLP...\t", str(datetime.now()))
        print("------------------------")
        print(self.nlp.pipe_names)
        print("------------------------")
        # df["nlp"] = [i for i in nlp.pipe(df["body"], batch_size=10, n_threads=6)]  # didn't really speed up anything
        # res = pd.DataFrame(df["body"].apply(nlp))
        # res.columns = ["nlp"]

        df["nlp"] = df["body"].apply(self.nlp)
        return df

    def spacy_ner(self, df):
        """ Use spacy CNN NER """
        # TODO should we try some of https://github.com/explosion/spaCy/tree/2d249a9502bfc5d3d2111165672d964b1eebe35e/bin/wiki_entity_linking        
        print("Starting NLP Coref\t", str(datetime.now()))
        print("------------------------")   
        print("Spacy NER pipeline:", self.nlp_pp.pipe_names)   
        print("------------------------")   
        df["nlp_resolved"] = df["nlp"].apply(lambda x: self.nlp_pp(x._.coref_resolved))
        print("Completed NLP by...\t", str(datetime.now()))
        return df

    def dbpedia_ner(self, df):
        """ Use DBPedia matching """
        
        # TODO we might be throwing out a lot of stuff in entity_linker (spacy_dbpedia_spotlight), maybe go check?
        
        print("Starting NLP Coref", str(datetime.now()))
        print("------------------------")   
        print("DBPedia NER pipeline:", self.nlp_dbp.pipe_names)   
        print("------------------------")
        def inner(x):
            return self.nlp_dbp(x)

        df["nlp_resolved"] = df["nlp"].apply(lambda x: inner(x._.coref_resolved))
        # res = pd.DataFrame(df["nlp"].apply(lambda x: inner(x._.coref_resolved)))
        # res.columns = ["nlp_resolved"]
        # res.append(df[])
        print("Completed NLP by...\t", str(datetime.now()))
        return df

    def find_most_common_entities(self, df, nlp_doc_colname:str, entity_type:str, df_name):
        """ Find most common entity for each article """
        # TODO put a similarity threshold in here somewhere!
        def find_most_common_entity(article_raw):
            article = article_raw[nlp_doc_colname]
            article_length = len(article.text)
            if article is None:
                return tuple()
            
            all_items = []  # mostly for debugging
            entity_counts = {}
            replacement_candidate = np.zeros(len(article.text)) -1

            for x in article.ents:
                if x.text.isspace():
                    continue
                text = x.text
                url = x.label_.split(" ")[0]
                dbpedia_labels = x.label_.split(" ")[1]
                all_items.append((text, url, dbpedia_labels)) 
                
                # resolve common disambiguitions
                if text.lower() == "washington":  # in doubt, go for location
                    standard_ner_ents = {i.text.lower(): i.label_ for i in article_raw["nlp"].ents}
                    if "washington" in [i.lower() for i in standard_ner_ents.keys()]:
                        if standard_ner_ents["washington"] == "GPE":
                            dbpedia_labels = "Location_State"
                elif text.lower() == "trudeau":  # in doubt, go for Justin
                    dbpedia_labels = "Justin Trudeau"

                if entity_type in dbpedia_labels:
                    # append last bit of url to make sure we don't get duplicates
                    last_url_tag = url.rfind("/")
                    entity_name = url[last_url_tag+1:].replace("_", " ")
                    if entity_name in entity_counts.keys():
                        entity_counts[entity_name] += 1
                    else:
                        entity_counts[entity_name] = 1
                    # set replacement candidate to index of corresponding entity in dict
                    ent_index = list(entity_counts).index(entity_name)
                    surface_form_start = article[x.start].idx
                    if x.end >= len(article):
                        surface_form_end = article_length
                    else:
                        surface_form_end = article[x.end].idx
                        # in case there is whitespace before next token, subtract
                        if article.text[article[x.end].idx-1] == " ":
                            surface_form_end -= 1
                    # TODO: possible bug here. If we have two candidates with the same names we for some reason do not replace them correcly
                    np.put(replacement_candidate, range(surface_form_start, surface_form_end), ent_index)
            
            if len(entity_counts) == 0:
                #TODO: changed this from:
                # return tuple()
                return ("None", 0, "".join(list(deepcopy(article.text))))

            mce = max(entity_counts)
            mce_list = list(mce)
            mce_val = entity_counts[mce]
            mce_idx = list(entity_counts).index(mce)

            in_a_row = False
            end = None
            start = None
            doc_len = article_length
            # TODO this list stuff isn't very efficient, could prolly be improved using pre-allocated arrays
            ner_resolved = list(deepcopy(article.text))
            for index, rep_idx in enumerate(replacement_candidate[::-1]):
                if (rep_idx != mce_idx) and (not in_a_row):
                    continue
                elif (rep_idx == mce_idx) and (not in_a_row):  # we have a mention of our enity
                    # we found start of surface form
                    in_a_row = True
                    end = doc_len - index
                    start = doc_len - index - 1
                elif ((rep_idx != mce_idx) or (index==doc_len-1)) and (in_a_row):
                    # we found end of surface form, replace
                    if (index==doc_len-1):
                        start -= 1
                    ner_resolved = ner_resolved[:start] + mce_list + ner_resolved[end:]
                    start = None
                    end = None
                    in_a_row = False
                elif (rep_idx == mce_idx) and in_a_row:
                    # we are still in surface form
                    start -= 1
            return (mce, mce_val, "".join(ner_resolved))

        df[[df_name, df_name+"_num", "ner_resolved_"+df_name]] = pd.DataFrame.from_records(
            df.apply(lambda x: find_most_common_entity(x), axis=1))  # apply to each row
        # df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)    
        return df

    def count_most_frequent(self, group):
        # current approach: most common of the most common - should be ok, I think
        items = [[i[0]] * i[1] for i in group if not (i is None)]
        items = list(chain.from_iterable(items))
        most_common = Counter(items).most_common(1)
        return most_common[0][0]  # get only the phrase for now

    def get_general_sentiment(self, df):
        df["g_sent"] = df["body"].apply(lambda x: self.gsa.get_general_sentiment(x))

        return df

    def get_target_sentiments(self, df_pp, targets):
        for target in targets:        
            df_pp["sents_" + target] = df_pp["ner_resolved_"+ target].apply(lambda x: list(self.nlp_nr(x, disable=["tokenizer","tagger","entity","ner"]).sents))

        def get_average_sentiment(sentences, target):
            sentiment_sum = 0
            count = 0

            trim_count = 0
            c_sentence_count = 0

            for sentence in sentences:
                sentiment_tulp = self.tsa.get_sentiment(sentence = str(sentence), target = str(target)) # Convert them to strings
                # The tuple contains (sentiment, sentence trimmed)
                sentiment = sentiment_tulp[0]
                if(sentiment is None):
                    # Nothing found in this sentence
                    continue
                else:
                    c_sentence_count += 1 #The sentence contains the target word
                    if(sentiment_tulp[1]):
                        # The sentence was trimmed
                        trim_count += 1

                    sentiment_sum += sentiment
                    count += 1

            if(count != 0):
                if(c_sentence_count == 0):
                    c_sentence_count = 1
                return (sentiment_sum/count, trim_count/c_sentence_count)
            else:
                return (0, trim_count)

        def get_covid_sentiment(sentences):
            sentiment_sum = 0
            count = 0

            trim_count = 0
            c_sentence_count = 0

            for sentence in sentences:
                sentence = str(sentence)
                sentence = sentence.lower()
                covid_list = ["corona", "covid"] #Note that if any substring contains this we will take that as target, thus these are not nescessary "coronavirus","covid-19", "covid19", 
                
                parsed_sentence = self.nlp_nr(sentence, disable=["parser","tagger","entity","ner"])
                
                found_covid = False

                for token in parsed_sentence:
                    tt = str(token.text)
                    if(found_covid):
                        break # Stop da loop

                    for word in covid_list:
                        loc = tt.find(word)
                        if(loc == -1):
                            # try next
                            continue
                        else:
                            # get the sentiment of the target word
                            sentiment_tulp = self.tsa.get_sentiment(sentence = str(sentence), target = str(tt)) # Convert them to strings
                            # The tuple contains (sentiment, sentence trimmed)
                             
                            sentiment = sentiment_tulp[0]
                            
                            if(sentiment is None):
                                # Nothing found in this sentence
                                continue
                            else:
                                c_sentence_count += 1 #The sentence contains the target word
                                if(sentiment_tulp[1]):
                                    # The sentence was trimmed
                                    trim_count += 1

                                sentiment_sum += sentiment
                                count += 1
                            
                            found_covid = True
                            break

            if(count != 0):
                if(c_sentence_count == 0):
                    c_sentence_count = 1
                return (sentiment_sum/count, trim_count/c_sentence_count)
            else:
                return (0, trim_count)

        # print(df_pp)

        # Get the average sentiment for each target
        for target in targets:
            df_pp[[target + "_sent",target + "_tls"]] = df_pp.apply(lambda x: pd.Series(get_average_sentiment(x["sents_"+target], x[target])), axis=1)
        
        
        df_pp[["c_sent","c_tls"]] = df_pp.apply(lambda x: pd.Series(get_covid_sentiment(x["sents_"+targets[0]])), axis=1)
        # Fill this with ones in order to be able to get the average in the end
        df_pp["c_count"] = 1
        return df_pp
    
    def show_problems(self, article, visualize=False):
        """ 
        show some of our current problems in sentiment analyses 
        @param artprint(tsa.get_sentiment("asdf","asdf"))icle: result of spacy nlp applied to the body of an article
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

def run_and_save(start_date, end_date, articles_per_period=None, max_length=None, with_sentiments=False, debug=False, delta_d=1):
    c_date = start_date

    # Name 
    run_name = "s_" + start_date.strftime("%d_%m_%Y") + "_e_" \
        + end_date.strftime("%d_%m_%Y") + "_app_" + str(articles_per_period) \
        + "_ml_" + str(max_length) + "_" + datetime.now().strftime("d_%d_%m_t_%H_%M")
    print(f"RUN NAME: {run_name}")
    if debug:
        run_name = "debug_run_" + datetime.now().strftime("d_%d_%m_t_%H_%M")

    folder_path = "data/"+run_name
    if not os.path.isdir(folder_path):
        os.mkdir(folder_path)
    else:
        print(folder_path)
        print("Already exists, stopping")
        return

    print("Start run from date: " + start_date.strftime("%d_%m_%Y") + " to date: " + end_date.strftime("%d_%m_%Y"))
    print("Articles per period: " + str(articles_per_period))
    print("Max Length per article: " + str(max_length))
    print("Delta day steps:",delta_d)

    start_time = time.process_time()

    NER = NamedEntityRecognizer(model_size="sm", with_sentiment=with_sentiments)  # model_size="lg")

    # Increase until we hit the last day
    while(c_date < end_date):
        try:

            print("Running day: " + str(c_date.strftime("%d_%m_%Y")) + " till " + str((c_date + timedelta(days=delta_d)).strftime("%d_%m_%Y")))
            print("Loading Data...\t", str(datetime.now()))
            if not debug:
                # df = read_data.get_body_df(
                #     start_date=c_date,
                #     end_date=c_date,
                #     articles_per_period=articles_per_period, #700,
                #     max_length=max_length
                # )
                df = read_data.get_representative_df(
                    start_date=c_date,
                    end_date=c_date+timedelta(days=delta_d),
                    max_length=max_length,
                    n_samples = articles_per_period*delta_d
                )

            if debug:

                df = pd.DataFrame({
                    "body": [
                    "The professor Yann LeCun hates it when his students have purple hair.",
                    "The professor Yann LeCun does not like it when his students have purple hair.",
                    "The professor Yann LeCun does likes it when his students have purple hair.",
                    "People in Greenland do not have the burden of the coronavirus, since they are isolated from the rest of the world they do not have to worry about the disease and all its negative implications.",
                    "People in Greenland have a great time, even though the coronavirus is terrible for the rest of the world they keep their positive vibe.",
                    "People in Greenland do have the burden of the coronavirus, since they are isolated from the rest of the world they do have to worry about the disease and all its negative implications."
                    ],
                    "publication_date": ["2020-04-05","2020-04-05","2020-04-05","2020-04-05","2020-04-05","2020-04-05"]
                })
                # df = pd.DataFrame({
                #     "body": [
                #     "New Zealand's two top public health bosses will be addressing New Zealand's to update New Zealand's on the latest Covid-19 developments. Director of Public Health Dr Caroline McElnay will join Director-General of Health Ashley Bloomfield...",
                #     "Deepika has a dog. She loves him. The movie star has always been fond of animals",
                #     "The short guy Donald Trump is the worst. He does not know how the world turns.",
                #     "The tall guy Donald Trump is the best.",
                #     "There were 23 more deaths linked to Covid-19 in the Netherlands, raising the total number of people who died in the country to 5,811. Public health agency RIVM said it also knew of another ten hospitalizations for the coronavirus disease.",
                #     "There were 23 more deaths linked to Covid-19 in the Netherlands, raising the total number of people who died in the country to 5,811 public health agency RIVM said it also knew of another ten hospitalizations for the coronavirus disease, there were 23 more deaths linked to Covid-19 in the Netherlands, raising the total number of people who died in the country to 5,811 public health agency RIVM said it also knew of another ten hospitalizations for the coronavirus disease.",
                #     "Coronavirus disease 2019 (COVID-19) is an infectious disease caused by severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2).[10] It was first identified in December 2019 in Wuhan, China, and has since spread globally, resulting in an ongoing pandemic.[11][12] As of 24 May 2020, more than 5.35 million cases have been reported across 188 countries and territories, resulting in more than 343,000 deaths. More than 2.14 million people have recovered."],
                    
                #     "publication_date": ["2020-04-05","2020-04-05","2020-04-05","2020-04-05","2020-04-05","2020-04-05","2020-04-05"]
                # })

            df = NER.spacy_preprocessing(df) # model_size="lg")
            if with_sentiments:
                df = NER.get_general_sentiment(df)
            # df.drop(columns=["body"], inplace=True) # Drop some columns to make some space
            df = NER.dbpedia_ner(df) #model_size="lg")
            df = NER.find_most_common_entities(df, "nlp_resolved", entity_type="Person", df_name =  "mc_p")  # entity "OfficeHolder" is quite nice, "Person" works as well
            df = NER.find_most_common_entities(df, "nlp_resolved", entity_type="Country", df_name =  "mc_c")

            df.drop(columns=["nlp"], inplace=True)

            if with_sentiments:
                targets = ["mc_p","mc_c"]
                df = NER.get_target_sentiments(df, targets = targets)

                for target in targets:
                    df.drop(columns=["sents_"+target], inplace=True)
                    df.drop(columns=["ner_resolved_"+target], inplace=True)

            # print(df.to_string())
            df.drop(columns=["nlp_resolved"], inplace=True) # Drop some columns to make some space


            # df.drop(columns=["nlp_resolved","ner_resolved"], inplace=True)
            # TODO check ner_resolved
            # print("aft ts",df.head())
            

            # # df = df[["publication_date", "most_common_1", "most_common_1_num", "t_sent", "c_sent"]]
            # df_most_common = NER.sum_period_most_common_entities(df)
            # print(df_most_common.head())


            print(df.head())
            if(delta_d == 1):
                file_name = c_date.strftime("%d_%m_%Y")
            else:
                file_name = c_date.strftime("%d_%m_%Y") + "_to_" +  (c_date + timedelta(days=delta_d)).strftime("%d_%m_%Y")
            df.to_csv(folder_path + "/" + file_name +".csv", index=False)
        except:
            print("Failed to do:",c_date.strftime("%d_%m_%Y"))
            print("Moving on")

        # Increase the day
        c_date += timedelta(days=delta_d)
    
    print("Multiple day run finished")
    elapsed_time = time.process_time() - start_time
    print("Elapsed time: " + str(round(elapsed_time,2)) + " seconds")
    


if __name__ == "__main__":

    if not os.path.isdir("src/experiments"):
        os.mkdir("src/experiments")

    if not os.path.isdir("src/logs"):
        os.mkdir("src/logs")

    # TODO: dates 2019-11-06 and 2020-01-01 throw errors
    # start_date=datetime.strptime("2019-11-01", "%Y-%m-%d")
    # end_date=datetime.strptime("2020-04-05", "%Y-%m-%d")

    # run_and_save(start_date, end_date, articles_per_period=None,     # df_most_common = get_viz_data.load_data("s_01_02_2020_e_05_04_2020_app_500_ml_300_d_26_05_t_15_20")
        
        # df_most_common = prepare_viz(df_most_common, mc_column="mc_p", mc_num_column="mc_p_num",
        #         sent_col="mc_p_sent", with_sentiment=True)
        # print(df_most_common.head())
        # start_date = df_most_common.publication_date.min()
        # end_date = df_most_common.publication_date.max()
        # visualize(df_most_common, start_date, end_date, "mc_p", "rolling_sent")

    start_date=datetime.strptime("2020-04-01", "%Y-%m-%d")
    end_date=datetime.strptime("2020-04-02", "%Y-%m-%d")

    run_and_save(start_date, end_date, articles_per_period=100,
         max_length=500, with_sentiments=True, debug=True, delta_d=1)
    
    # start_date=datetime.strptime("2020-03-01", "%Y-%m-%d")
    # end_date=datetime.strptime("2020-04-05", "%Y-%m-%d")


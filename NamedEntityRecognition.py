""" Named Entity Recognition """

import read_data
from datetime import datetime, timedelta
import spacy  # the spacy nets have been trained on OntoNotes 5.0
from spacy import displacy
from collections import Counter
import pandas as pd
from itertools import chain
import visualization


def most_common_entities(df, nlp_doc_colname:str, n_common:int):
    """ 
    get most common entities for a series of articles on a certain date 
    @param df: data frame containing our articles
    @param nlp_doc_colname: name of column containing nlp-processed documents
    @param n_common: number of most frequent to keep
    """
    most_common = []
    for d in df[nlp_doc_colname]:
        items = [x.text for x in d.ents]
        most_common += Counter(items).most_common(n_common)
    colnames = []
    for i in range(1, n_common+1):
        colnames.append(f"most_common_{i}")
        colnames.append(f"most_common_{i}_num")
    df_most_common = pd.DataFrame.from_records(most_common, columns=colnames)
    return df_most_common


# TODO include functionality for counting most frequent Person, Location, etc.
def count_most_frequent(group):
    # current approach: most common of the most common - should be ok, I think
    items = [[i[0]] * i[1] for i in group if not (i is None)]
    items = list(chain.from_iterable(items))
    most_common = Counter(items).most_common(1)
    return most_common[0][0]  # get only the phrase for now


if __name__ == "__main__":
    print("Loading the data")
    df = read_data.get_body_df(
        start_date=datetime.strptime("2020-03-29", "%Y-%m-%d"),
        end_date=datetime.strptime("2020-04-06", "%Y-%m-%d"),
        articles_per_period=10,
    )

    print("Starting NLP")

    nlp = spacy.load("en_core_web_sm")  # "eng_core_web_lg" for better but slower results
    df["nlp"] = [nlp(doc) for doc in df.body]  # might be a lot faster if we merge all articles of a day into one document?

    df_most_common = most_common_entities(df, "nlp", 1)
    df_most_common["publication_date"] = df["publication_date"]
    df_most_common = df_most_common.groupby(by=["publication_date", "most_common_1"])
    df_most_common = df_most_common.agg(sum).sort_values(by="most_common_1_num", ascending=False).reset_index()
    df_most_common = df_most_common.groupby(by=["publication_date"])
    df_most_common = df_most_common.head(10)

    print("Starting visualization")

    visualization.animate_NER(df_most_common)

# TODO example: doesn't understand that Johnson is same as Boris Johnson
# TODO for word vectors & similarity, we might have to train new model to catch stuff like "Coronavirus"
# TODO basically: tailor model to our data set
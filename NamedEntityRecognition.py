import read_data
from datetime import datetime, timedelta
import spacy  # the spacy nets have been trained on OntoNotes 5.0
from spacy import displacy
from collections import Counter
import pandas as pd
from itertools import chain
import seaborn as sns
import matplotlib.pyplot as plt

""" Named Entity Recognition """

# TODO feels like our data is sorted newest to oldest... is that right?

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
    # df = pd.concat((df, df_most_common), join="outer", axis=1)
    return df_most_common


# TODO: the way we're loading stuff rn is terribly inefficient! 
# we should load each article, then see if it fits anywhere!
def load_articles_per_day(articles_per_period, period_start, period_end):
    """ 
    load a given number of articles for each day in period 
    """
    df = pd.DataFrame()
    period_length = (period_end - period_start).days
    for i in range(period_length):
        start_date = period_start + timedelta(days=i)
        end_date = start_date + timedelta(days=1)
        df_temp = read_data.get_body_df(articles_per_period, start_date=start_date, end_date=end_date)
        # TODO I think this might be a slow way of doing it... 
        df = pd.concat((df, df_temp), axis=0, ignore_index=True)
    return df

# TODO include functionality for counting most frequent Person, Location, etc.
def count_most_frequent(group):
    # current approach: most common of the most common - should be ok, I think
    items = [[i[0]] * i[1] for i in group if not (i is None)]
    items = list(chain.from_iterable(items))
    most_common = Counter(items).most_common(1)
    return most_common[0][0]  # get only the phrase for now


df = load_articles_per_day(
    articles_per_period=10,
    period_start=datetime.strptime("2020-04-01", "%Y-%m-%d"),
    period_end=datetime.strptime("2020-04-06", "%Y-%m-%d")
)

nlp = spacy.load("en_core_web_sm")  # "eng_core_web_lg"
df["nlp"] = [nlp(doc) for doc in df.body]


# TODO: this is what we want; make use of it!!
df_most_common = most_common_entities(df, "nlp", 1)
df_most_common["publication_date"] = df["publication_date"]
df_most_common = df_most_common.groupby(by=["publication_date", "most_common_1"])
df_most_common = df_most_common.agg(sum).sort_values(by="most_common_1_num", ascending=False).reset_index()
df_most_common = df_most_common.groupby(by=["publication_date"])
df_most_common = df_most_common.head(3)

sns.barplot(data=df_most_common, x="most_common_1_num", y="most_common_1")
plt.show()

# TODO animate the barplot by date dimension


# TODO create visualization module
# TODO create animated plot for change over time
# # show in browser
# test_id = 5
# displacy.serve(nlp(str(articles[test_id])), style='ent')
# displacy.serve(nlp(str(articles[test_id])), style='dep', options = {'distance': 120})

# TODO get most important entities from a batch of documents
# TODO get top entities for each day, create one of these sweet stats videos on change over time

# TODO example: doesn't understand that Johnson is same as Boris Johnson

# TODO for word vectors & similarity, we might have to train new model to catch stuff like "Coronavirus"


### Animation
# https://stackoverflow.com/questions/42056347/animated-barplot-in-python
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
plt.style.use('seaborn-pastel')

animation_start_date = min(df_most_common["publication_date"])
def init():
    df_filtered = df_most_common.loc[df_most_common["publication_date"]==animation_start_date,:]
    y = df_filtered["most_common_1_num"].tolist()
    return y

fig, ax = plt.subplots()
n_frames = len(df["publication_date"].unique()) #Number of frames
x = list(range(0,3))
barcollection = plt.bar(x, init())#, tick_label=["a", "b", "c"])
barcollection.set_label(["a", "b", "c"])

def animate(i):
    new_date = animation_start_date + timedelta(days=i)
    df_filtered = df_most_common.loc[df_most_common["publication_date"]==new_date,:]
    labels = df_filtered["most_common_1"].tolist()
    # TODO set labels
    y = df_filtered["most_common_1_num"].tolist()
    for i, b in enumerate(barcollection):
        b.set_height(y[i])

anim = FuncAnimation(fig, animate, repeat=True, blit=False, frames=n_frames-1, interval=100)
plt.show()


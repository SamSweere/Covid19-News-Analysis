import jsonlines
from datetime import datetime, timedelta
import pandas as pd
from copy import deepcopy


def filter_articles(n_articles=None, source_name=None, start_date=None, end_date=None, articles_per_period=None, max_length=None):
    """
    filter articles according to specifications
    our data seems to be sorted newest to oldest -> retrieving new data is much faster
    """
    date_of_publication = []
    articles = []
    counter = 0
    articles_per_period_counter = 0
    current_period = None
    with jsonlines.open("data/aylien-covid-news.jsonl") as f:
        for line in f:
            t = line["published_at"].split(" ")[0]
            d_o_p = datetime.strptime(t, "%Y-%m-%d")
            if current_period is None:
                current_period = d_o_p
            if not (n_articles is None):
                if(counter  >= n_articles):
                    break
            if not (articles_per_period is None):
                if articles_per_period_counter == articles_per_period:  # article limit reached
                    if d_o_p != current_period:  # new period
                        current_period = d_o_p
                        articles_per_period_counter = 0
                    else:
                        continue
            if not (source_name is None):
                if line["source"]["name"] != source_name:
                    continue
            if not (start_date is None):
                if start_date > d_o_p:
                    break  # since our data is ordered newest to oldest, we can do this
            if not (end_date is None):
                if end_date < d_o_p:
                    continue
            if len(line["body"]) > max_length:
                continue
            articles.append(line)
            date_of_publication.append(deepcopy(d_o_p))
            counter += 1
            articles_per_period_counter += 1
    df = pd.DataFrame({
        "article": articles,
        "publication_date": date_of_publication
    })
    return df

def available_metadata():
    """print tags for available metadata"""
    with jsonlines.open("aylien-covid-news.jsonl") as f:
        for line in f:
            return line.keys()


def get_body_df(n_articles=None, source_name=None, start_date=None, end_date=None, articles_per_period=None, max_length=None):
    df = filter_articles(n_articles, source_name, start_date, end_date, articles_per_period, max_length)
    df["body"] = [i["body"] for i in df["article"]]
    return df[["body", "publication_date"]]


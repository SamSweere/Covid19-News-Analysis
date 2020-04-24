import jsonlines
from datetime import datetime
import pandas as pd


def filter_articles(n_articles, source_name=None, start_date=None, end_date=None):
    """filter articles according to specifications"""
    date_of_publication = []
    articles = []
    counter = 0
    with jsonlines.open("aylien-covid-news.jsonl") as f:
        for line in f:
            t = line["published_at"].split(" ")[0]
            d_o_p = datetime.strptime(t, "%Y-%m-%d")
            if(counter  >= n_articles):
                break
            if not (source_name is None):
                if line["source"]["name"] != source_name:
                    continue
            if not (start_date is None):
                if start_date > d_o_p:
                    continue
            if not (end_date is None):
                if end_date < d_o_p:
                    continue
            articles.append(line)
            date_of_publication.append(d_o_p)
            counter += 1
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


def get_body_df(n_articles, source_name=None, start_date=None, end_date=None):
    df = filter_articles(n_articles, source_name, start_date, end_date)
    df["body"] = [i["body"] for i in df["article"]]
    return df[["body", "publication_date"]]


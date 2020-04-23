import jsonlines
from datetime import datetime
import pandas as pd


def filter_articles(n_articles, source_name=None, start_date=None, end_date=None):
    """filter articles according to specifications"""
    articles = []
    counter = 0
    with jsonlines.open("aylien-covid-news.jsonl") as f:
        for line in f:
            if(counter  >= n_articles):
                break
            if not (source_name is None):
                if line["source"]["name"] != source_name:
                    continue
            if not (start_date is None):
                # TODO some nice date comparison
                t = line["published_at"].split(" ")[0]
                date = datetime.strptime(t, "%Y-%m-%d")
                if start_date > date:
                    continue
            if not (end_date is None):
                t = line["published_at"].split(" ")[0]
                date = datetime.strptime(t, "%Y-%m-%d")
                if end_date < date:
                    continue
            articles.append(line)
            counter += 1

    return articles

def available_metadata():
    """print tags for available metadata"""
    with jsonlines.open("aylien-covid-news.jsonl") as f:
        for line in f:
            return line.keys()


def get_body_df(n_articles, source_name=None, start_date=None, end_date=None):
    articles = filter_articles(n_articles, source_name, start_date, end_date)
    bodies = [i["body"] for i in articles]
    df = pd.DataFrame({
        "body": bodies
    })
    return df


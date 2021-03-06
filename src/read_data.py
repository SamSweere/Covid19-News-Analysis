import jsonlines
from datetime import datetime, timedelta
import pandas as pd
from copy import deepcopy
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.dates import (YEARLY, DateFormatter,
                              rrulewrapper, RRuleLocator, drange)
np.random.seed(0)


def filter_articles(n_articles=None, source_name=None, start_date=None, end_date=None, articles_per_period=None, max_length=None):
    """
    filter articles according to specifications
    our data seems to be sorted newest to oldest -> retrieving new data is much faster
    """
    if not max_length:
        max_length = np.inf
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
                # Crop the news message
                line["body"] = line["body"][:max_length]
                # continue
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


def get_representative_df(n_samples, start_date, end_date, max_length=None):
    """ get a df representative of a given time period """
    counter = 0
    with jsonlines.open("data/aylien-covid-news.jsonl") as f:
        for line in f:
            t = line["published_at"].split(" ")[0]
            d_o_p = datetime.strptime(t, "%Y-%m-%d")
            if d_o_p < start_date:
                break
            if (d_o_p >= start_date) and (d_o_p <= end_date):
                counter += 1

    if(n_samples is None):
        n_samples = counter

    index_list = sorted(np.unique([np.random.randint(0, counter) for i in range(n_samples)]), reverse=True)

    bodies = []
    date_of_publication = []
    counter = 0

    if not max_length:
        max_length = np.inf
    
    with jsonlines.open("data/aylien-covid-news.jsonl") as f:
        for line in f:
            if not index_list:
                break
            t = line["published_at"].split(" ")[0]
            d_o_p = datetime.strptime(t, "%Y-%m-%d")
            if d_o_p < start_date:
                break
            if (d_o_p >= start_date) and (d_o_p <= end_date):
                if counter == index_list[-1]:
                    index_list.pop()
                    date_of_publication.append(d_o_p)
                    if len(line["body"]) > max_length:
                        # Crop the news message
                        line["body"] = line["body"][:max_length]
                    bodies.append(line["body"])
            
                counter += 1
    
    df = pd.DataFrame({
        "body": bodies,
        "publication_date": date_of_publication
    })
    # count length of time period
    # random shuffle indexes within period
    # take top n
    # get representative sample
    return df

def get_count_df():
    # count articles per day
    date_of_publication = []
    all_counts = []
    all_n_char = []
    prev_dop = None
    count = 0
    n_char = 0
    with jsonlines.open("data/aylien-covid-news.jsonl") as f:
        for line in f:
            t = line["published_at"].split(" ")[0]
            d_o_p = datetime.strptime(t, "%Y-%m-%d")
            if prev_dop is None:
                prev_dop = d_o_p
                count = 1
                n_char = len(line["body"])
            elif prev_dop != d_o_p:
                date_of_publication.append(prev_dop)
                all_counts.append(count)
                all_n_char.append(n_char)
                prev_dop = d_o_p
                count = 1
                n_char = len(line["body"])
            else:
                count += 1
                n_char += len(line["body"])
    df = pd.DataFrame({
        "publication_date": date_of_publication,
        "num_articles": all_counts,
        "num_characters": all_n_char
    })
    df.to_csv("data/df_counts.csv", index=False)


if __name__ == "__main__":
    # Plot count df for report
    
    df = pd.read_csv("data/df_counts.csv")
    df["publication_date"] = df["publication_date"].apply(lambda x: datetime.strptime(x, "%Y-%m-%d"))
    df.sort_values(by=["publication_date"], inplace=True)
    mask = df.publication_date.apply(lambda x: str(x.year) == "2020")
    df = df[mask]

    prev_date = None
    def format_date(x):
        global prev_date
        if prev_date is None:
            prev_date = x
            return str(prev_date)[5:10]
        if prev_date.week != x.week:
            prev_date = x
            return str(prev_date)[5:10]
        return ""

    df["day_of_week"] = df.publication_date.apply(lambda x: format_date(x))

    p = sns.barplot(data=df, x="publication_date", y="num_articles")
    labels=df.day_of_week.unique()
    p.set_xticklabels(labels=df.day_of_week, rotation=90) #, ha='right')
    plt.xlabel("publication_date")
    plt.tight_layout()
    plt.savefig("src/figures/articles_per_day.png", dpi=500)
    plt.show()

    # fig, ax = plt.subplots(figsize=(10, 6))
    # ax.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
    # ax.bar(df["dates"], df['num_characters'], width=25, align='center')
    # plt.title("Number of characters per day")
    # plt.savefig("src/figures/characters_per_day.png", dpi=500)
    # plt.show()

    pass
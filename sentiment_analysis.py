import read_data
from datetime import datetime, timedelta
import spacy  # the spacy nets have been trained on OntoNotes 5.0
import pandas as pd

if __name__ == "__main__":
    print("Loading the data")
    df = read_data.get_body_df(
        start_date=datetime.strptime("2020-03-29", "%Y-%m-%d"),
        end_date=datetime.strptime("2020-04-06", "%Y-%m-%d"),
        articles_per_period=10,
    )

    print("Starting NLP")

    # nlp = spacy.load("en_core_web_sm")

    # df["nlp"] = [nlp(doc) for doc in df.body]  # might be a lot faster if we merge all articles of a day into one document?
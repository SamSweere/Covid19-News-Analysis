import read_data
from datetime import datetime
import spacy

""" Named Entity Recognition """

# load data
start_date = datetime.strptime("2020-03-10", "%Y-%m-%d")
end_date = datetime.strptime("2020-04-20", "%Y-%m-%d")
df = read_data.get_body_df(
    n_articles=1000, start_date=start_date, end_date=end_date)


from spacy import displacy
from collections import Counter
# NLP does literally everything for us
nlp = spacy.load("en_core_web_sm")

all_docs = read_data.get_body_df(10).body
articles = [nlp(doc) for doc in all_docs]

# show in browser
displacy.serve(nlp(str(articles[3])), style='ent')
displacy.serve(nlp(str(articles[3])), style='dep', options = {'distance': 120})

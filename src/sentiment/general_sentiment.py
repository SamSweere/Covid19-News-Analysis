import numpy as np
import spacy
import torch

class GeneralSentimentAnalyzer:
    def __init__(self):
        model_path = "models/xlnet_sentiment"
        print("Loading model:",model_path)
        self.s_nlp = spacy.load(model_path)

    def get_general_sentiment(self, article):
        article = str(article)
        sentiment = self.s_nlp(article).doc.cats.items()
        pos_sent = list(sentiment)[0][1]

        sent = (pos_sent - 0.5)*2
        return round(sent,2)

# gsa = GeneralSentimentAnalyzer()
# print(gsa.get_general_sentiment("Deepika has a dog. She loves him. The movie star has always been fond of animals"))
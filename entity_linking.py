import spacy
import requests
import spacy_dbpedia_spotlight

# use your model
nlp = spacy.load('en_core_web_sm') #, disable=["ner"])

# sentencizer (otherwise our api calls will be too long)
sentencizer = nlp.create_pipe("sentencizer")
nlp.add_pipe(sentencizer, first=True)

# pass nlp as parameter
# spacy_dbpedia_spotlight.load('en', nlp)

doc = nlp('The president of USA is calling Boris Johnson to decide what to do about coronavirus- This is a second sentence')
# TODO the final label stuff is not gonna work for spacy < 2.2
print("Entities", [(ent.text, ent.label_) for ent in doc.ents])
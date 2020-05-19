
# # Load your usual SpaCy model (one of SpaCy English models)
# import spacy
# import neuralcoref

# nlp = spacy.load('en_core_web_sm')

# # Add neural coref to SpaCy's pipe
# neuralcoref.add_to_pipe(nlp)

# # You're done. You can now use NeuralCoref as you usually manipulate a SpaCy document annotations.
# doc = nlp(u'My sister has a dog. She loves him.')

# doc._.has_coref
# doc._.coref_clusters

import spacy
import spacy_dbpedia_spotlight

# use your model
nlp = spacy.load('en_core_web_sm', disable=["ner"])
# pass nlp as parameter
spacy_dbpedia_spotlight.load('en', nlp)
doc = nlp('The president of USA is calling Boris Johnson to decide what to do about coronavirus')
print("Entities", [(ent.text, ent.label_, ent.kb_id_) for ent in doc.ents])


import spacy
import spacy_dbpedia_spotlight

# use your model
nlp = spacy.load('en_core_web_lg')
# pass nlp as parameter
spacy_dbpedia_spotlight.load('en', nlp)
doc = nlp('The president of USA is calling Boris Johnson to decide what to do about coronavirus')
print("Entities", [(ent.text, ent.label_, ent.kb_id_) for ent in doc.ents])


# Do everything by hand
# TODO entity linker seems to work just fine, not sure what the prolem is...
import requests
base_url = "http://api.dbpedia-spotlight.org/"
language_code = "en"
r = requests.get(f'{base_url}{language_code}/annotate', headers={'accept': 'application/json'}, params={'text': doc.text})
print(r.text)

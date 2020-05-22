import spacy
# spacy.require_gpu()

# Use sm for now, lg is hella big
nlp = spacy.load("en_core_web_sm")

tokens = nlp("coronavirus is covid-19 is covid is corona")

for token in tokens:
    print(token.text, token.has_vector, token.vector_norm, token.is_oov)
    print("Vector:", token.vector)
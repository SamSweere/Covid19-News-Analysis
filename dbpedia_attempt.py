import spacy

# any nlp you want
nlp = spacy.blank('en_core_weg_sm')
# create the pipe component, the dict argument is optional
entity_annotator = nlp.create_pipe('annotate_dbpedia_spotlight', {'language_code':'it'})
# add on your fancy pipeline with options like `first`
nlp.add_pipe(entity_annotator, first=True)
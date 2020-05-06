
# Load your usual SpaCy model (one of SpaCy English models)
import spacy
import neuralcoref

nlp = spacy.load('en_core_web_sm')

# Add neural coref to SpaCy's pipe
neuralcoref.add_to_pipe(nlp)

# You're done. You can now use NeuralCoref as you usually manipulate a SpaCy document annotations.
doc = nlp(u'My sister has a dog. She loves him.')

doc._.has_coref
doc._.coref_clusters
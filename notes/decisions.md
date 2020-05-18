parameter for limiting document size ( might make for fair comparison)?
Use DBPedia linker for real world entity recognition
Run real world entity linker on ner-coref resolved text (second separate pipeline)
We can now set more specific entity tags
We could restrict results by adding query to api call
Of all entity thingys, we take the ones that start with 'DBPedia:'
Currently, we take all entities we get and we check if our query string is in there
We have overlapping entiies (e. g. Washington D.C), we resolve by taking larger one!
Fight duplicate entities ("Trump" vs "Donald Trump")
We see that e.g. the name "Marco Sylvester" in one of the articles triggers finding Looney Toons character "Sylvester the Cat" - in general, dbpedia tends to look for the one prominant instance of any entity (if there is a "Tom", it's probably "Tom Hanks"). This is sometimes too sensitive but balanced out by the law of large numbers.
We take the last bit of the dbpedia URL to make sure that same entities with different surface form are recognized to correspond to the same real world entity.
We get a lot of problems with disambiguition around "Washington" in particular. Washington, D.C. is fine, but "Washington" on its own is mostly classified as the first US president by dbpedia (The standard NER classifies it as GPE, so that's not the problem).
Set greedyness of neuralcoref to 4 after seeing that otherwise it tends to replace too much.
We see that descriptive references are resolved as real ones: "Donald Trump's aide xyz" will be seen as a mention of Donald Trum as well. (Example: Washingon state Gov. Jay Inslee is seen as one entity by neuralcoref, but as mulitple by NER and dbpedia). 
We see that sometimes spacy NER has better results than DBPedia, in particular in cases of disambiguition.
For now we manually exclude the Washington case bc it seems the only prominent disambiguition case that really interferes with finding Person entities.
We find that a number of documents has double whitespaces. Often enough these are wrongly recognized as a token. We make sure to remove them in the early stages of the pipeline.
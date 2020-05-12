parameter for limiting document size ( might make for fair comparison)?
Use DBPedia linker for real world entity recognition
Run real world entity linker on ner-coref resolved text (second separate pipeline)
We can now set more specific entity tags
We could restrict results by adding query to api call
Of all entity thingys, we take the ones that start with 'DBPedia:'
Currently, we take all entities we get and we check if our query string is in there
We have overlapping entiies (e. g. Washington D.C), we resolve by taking larger one!
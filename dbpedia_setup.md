# key spacy wrapper file:
TextMiningCourse/tm_venv/lib/python3.6/site-packages/spacy_dbpedia_spotlight/entity_linker.py

# docker image
https://github.com/dbpedia-spotlight/spotlight-docker


# access
https://stackoverflow.com/questions/50735033/how-to-use-dbpedia-spotlight-docker-image

## example:
curl "http://localhost:2222/rest/annotate?text=President%20Michelle%20Obama%20called%20Thursday%20on%20Congress%20to%20extend%20a%20tax%20break%20for%20students%20included%20in%20last%20year%27s%20economic%20stimulus%20package,%20arguing%20that%20the%20policy%20provides%20more%20generous%20assistance.&confidence=0.2&support=20" -H "Accept:application/json"

import spacy
import requests

from spacy.tokens import Span

supported_languages = ['en', 'de', 'es', 'fr', 'it', 'nl', 'pt', 'ru']

class DbpediaLinker(object):
    # base_url = 'http://api.dbpedia-spotlight.org'
    ### custom base url
    base_url = "http://localhost:2222/rest"

    def __init__(self, language_code, nlp=None):
        if language_code not in supported_languages:
            raise ValueError(f'Linker not available in {language_code}. Choose one of {supported_languages}')
        self.language_code = language_code

        if not nlp:
            nlp = spacy.blank(language_code)
        
        if 'ner' not in nlp.pipe_names:
            ner = nlp.create_pipe('ner')
            nlp.add_pipe(ner)
            nlp.begin_training()
        
        nlp.entity.add_label('DBPEDIA_ENT')

        def annotate_dbpedia_spotlight(doc):
            self._annotate_dbpedia_spotlight(doc)
            return doc
        
        nlp.add_pipe(annotate_dbpedia_spotlight)
        self.nlp = nlp


    def _annotate_dbpedia_spotlight(self, doc, debug=False, chunk_id=0):

        # TODO doc.text can easily be too long for api call!
        # TODO split document in chunks of max 2500?
        # url_string = f'{self.base_url}{self.language_code}/annotate'
        url_string = f"{self.base_url}/annotate"
        counter = 0
        while(counter < 10):
            response = requests.get(url_string, headers={'accept': 'application/json'}, params={'text': doc.text})
            if response.status_code == 200:
                break
            counter += 1

        # if response status is still error: just don't do entity recognition
        if response.status_code == 403:
            print(f"Error for doc '{doc.text[0:30]}...'")
            return doc

        response.raise_for_status()
        data = response.json()
        if debug:
            print(data)

        ents_data = []
        ents_start_end_len = []

        for ent in data.get('Resources', []):
            start_ch = int(ent['@offset'])
            end_ch = int(start_ch + len(ent['@surfaceForm']))
            length = end_ch-start_ch
            start_end_len = (start_ch, end_ch, length)
            
            # make sure it does not overlap with any we've found already
            # TODO we are still getting Washington the person a lot, we might need to look into that
            conflict = True
            while conflict:
                to_delete = []
                dont_process_current = False
                for i, (s, e, l) in enumerate(ents_start_end_len):
                    if (start_ch < s) and (end_ch > s):  # I don't think this can happen
                        if length > l:
                            to_delete.append(i)
                        else:
                            dont_process_current=True
                    elif (start_ch < e) and (end_ch > e):
                        if length > l:
                            to_delete.append(i)
                        else:
                            dont_process_current=True
                    elif (start_ch < s) and (end_ch > e):
                        to_delete.append(i)
                    elif (start_ch >= s) and (end_ch <= e):
                        dont_process_current = True
                if dont_process_current:
                    break
                for i in to_delete:
                    ents_data.pop(i)
                    ents_start_end_len.pop(i)
                if len(to_delete) == 0:
                    conflict = False
            if dont_process_current:
                continue


            ent_kb_id = ent['@URI']
            # TODO look at '@types' and choose most relevant?
            # TODO we seem to be getting overlapping entities here every now and then... why??
            ent_type = "_".join([i[8:] for i in ent["@types"].split(",") if i.startswith("DBpedia")])
            if ent_type is None:
                ent_type = 'DBPEDIA_ENT'
            # put both ent_kb_id and ent_type into the same variable bc we are lacking
            # ent_kb_id in spacy < 2.2
            ent_type = f"{ent_kb_id} {ent_type}"
            span = doc.char_span(start_ch, end_ch, ent_type, ent_kb_id)
            if not (span is None):
                ents_data.append(span)
                ents_start_end_len.append(start_end_len)
        
        # print([(i, i.start, i.end) for i in ents_data])
        # print(ents_data)
        doc.ents = list(doc.ents) + ents_data
        return doc

from sentiment import target_based_sentiment


from NamedEntityRecognition import *

start_time = time.process_time()

# df = pd.DataFrame({
#     "body": ["With all the havoc it's wreaking across the globe, the coronavirus outbreak is naturally having an impact on couples and their relationships.","A tiger at New York's Bronx Zoo has tested positive for COVID-19, the institution said Sunday, and is believed to have contracted the virus from a caretaker who was asymptomatic at the time. The four-year-old Malayan tiger named Nadia along with her sister Azul, two Amur tigers and three African lio"],
#     "publication_date": ["2020-04-05","2020-04-05"]
# })

df = pd.DataFrame({
    "body": ["Deepika has a dog. She loves him. The movie star has always been fond of animals",
    "The short guy Donald Trump is the worst. He does not know how the world turns."],
    "publication_date": ["2020-04-05","2020-04-05"]
})

NER = NamedEntityRecognizer()

df_pp = NER.spacy_preprocessing(df, model_size="sm") # model_size="lg")
df_pp = NER.dbpedia_ner(df_pp, model_size="sm") #model_size="lg")


del df
df_pp = NER.find_most_common_entities(df_pp, "nlp_resolved", entity_type="Person")  # entity "OfficeHolder" is quite nice, "Person" works as well
    



# print(df_pp.iloc[1]["nlp"]._.coref_clusters)
# print(df_pp.iloc[1]["nlp"]._.coref_resolved)
# print(df_pp.iloc[0]["nlp"])

# print(df_pp.iloc[1]["nlp_resolved"])
# print(df_pp.iloc[1]['annotate_dbpedia_spotlight'])

nlp_nr = spacy.load("en_core_web_sm")
# doc = nlp("text goes here", tokenizer = False, parser=False, tagger=False, entity=False)



# print(list(df_pp.iloc[0]["ner_resolved"].sents))

df_pp["sents"] = df_pp["ner_resolved"].apply(lambda x: list(nlp_nr(x, disable=["tokenizer","tagger","entity","ner"]).sents))


print(df_pp.head())
# print(df_pp.iloc[0]["sents"])
# print(df_pp.iloc[0]["sents"][0])
# print(df_pp.iloc[0]["most_common_1"])



tsa = target_based_sentiment.TargetSentimentAnalyzer()  


def get_average_sentiment(sentences, target):
    print(sentences)
    print(target)
    print()
    sentiment_sum = 0
    count = 0

    for sentence in sentences:
        sentiment = tsa.get_sentiment(sentence = str(sentence), target = str(target)) # Convert them to strings

        if(sentiment is None):
            # Nothing found in this sentence
            continue
        else:
            sentiment_sum += sentiment
            count += 1

    if(count != 0):
        return sentiment_sum/count
    else:
        return 0


# df_pp["most_common_1"]
# df_pp["ner_resolved"]

# df

df_pp["sentiment"] = df_pp.apply(lambda x: get_average_sentiment(x["sents"], x["most_common_1"]), axis=1)

print(df_pp.head())


# print(tsa.get_sentiment("The tall guy Philip Frederics is the worst.","Philip Frederics"))

elapsed_time = time.process_time() - start_time
print("Elapsed time: " + str(round(elapsed_time,2)) + " seconds")
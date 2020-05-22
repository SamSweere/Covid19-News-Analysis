from NamedEntityRecognition import *

start_time = time.process_time()

# df = pd.DataFrame({
#     "body": ["With all the havoc it's wreaking across the globe, the coronavirus outbreak is naturally having an impact on couples and their relationships.","A tiger at New York's Bronx Zoo has tested positive for COVID-19, the institution said Sunday, and is believed to have contracted the virus from a caretaker who was asymptomatic at the time. The four-year-old Malayan tiger named Nadia along with her sister Azul, two Amur tigers and three African lio"],
#     "publication_date": ["2020-04-05","2020-04-05"]
# })

df = pd.DataFrame({
    "body": ["Deepika has a dog. She loves him. The movie star has always been fond of animals",
    "A tiger at New York's Bronx Zoo has tested positive for COVID-19, the institution said Sunday, and is believed to have contracted the virus from a caretaker who was asymptomatic at the time. The four-year-old Malayan tiger named Nadia along with her sister Azul, two Amur tigers and three African lio"],
    "publication_date": ["2020-04-05","2020-04-05"]
})

NER = NamedEntityRecognizer()

df_pp = NER.spacy_preprocessing(df, model_size="sm") # model_size="lg")


# df_pp = NER.dbpedia_ner(df_pp,model_size="sm") #model_size="lg")

# df_pp = NER.find_most_common_entities(df_pp, "nlp_resolved", entity_type="Person")  # entity "OfficeHolder" is quite nice, "Person" works as well
# df_pp = df_pp[["publication_date", "most_common_1", "most_common_1_num"]]
# df_most_common = NER.sum_period_most_common_entities(df_pp)
# df_most_common = NER.fill_entity_gaps(df_most_common)
# df_most_common = NER.cum_sum_df(df_most_common)
# df_most_common = NER.select_most_common_per_period(df_most_common)

elapsed_time = time.process_time() - start_time
print("Elapsed time: " + str(round(elapsed_time,2)) + " seconds")


# print(df_pp.iloc[0]["nlp"]._.coref_clusters)
# print(df_pp.iloc[0]["nlp"]._.coref_resolved)
print(df_pp.iloc[0]["nlp"])

from sentiment import target_based_sentiment


from NamedEntityRecognition import *

start_time = time.process_time()

# df = pd.DataFrame({
#     "body": ["With all the havoc it's wreaking across the globe, the coronavirus outbreak is naturally having an impact on couples and their relationships.","A tiger at New York's Bronx Zoo has tested positive for COVID-19, the institution said Sunday, and is believed to have contracted the virus from a caretaker who was asymptomatic at the time. The four-year-old Malayan tiger named Nadia along with her sister Azul, two Amur tigers and three African lio"],
#     "publication_date": ["2020-04-05","2020-04-05"]
# })

df = pd.DataFrame({
    "body": ["Deepika has a dog. She loves him. The movie star has always been fond of animals",
    "The short guy Donald Trump is the worst. He does not know how the world turns.",
    "The tall guy Donald Trump is the best."],
    "publication_date": ["2020-04-05","2020-04-05","2020-04-05"]
})

NER = NamedEntityRecognizer()

df = NER.spacy_preprocessing(df, model_size="sm") # model_size="lg")

df = df.drop(columns=["body"]) # Drop some columns to make some space

df = NER.dbpedia_ner(df, model_size="sm") #model_size="lg")
# Cleanup df_pp by removing nlp
df = df.drop(columns=["nlp"])
    
df = NER.find_most_common_entities(df, "nlp_resolved", entity_type="Person")  # entity "OfficeHolder" is quite nice, "Person" works as well

df = NER.get_target_sentiments(df, model_size="sm")


print(df.head())

df = df[["publication_date", "most_common_1", "most_common_1_num", "sentiment"]]
df_most_common = NER.sum_period_most_common_entities(df)
df_most_common = NER.fill_entity_gaps(df_most_common)
df_most_common = NER.cum_sum_df(df_most_common)
df_most_common = NER.select_most_common_per_period(df_most_common)

# df_most_common.to_csv("src/logs/df_most_common"+str(datetime.now())+".csv")
print(df_most_common)
NER.visualize(df_most_common, start_date, end_date)

elapsed_time = time.process_time() - start_time
print("Elapsed time: " + str(round(elapsed_time,2)) + " seconds")
categories = ['alt.atheism', 'soc.religion.christian'] 
 
#Loading the data set - training data.
from sklearn.datasets import fetch_20newsgroups
import matplotlib.pyplot as plt
  
newsgroups_train = fetch_20newsgroups(subset='train', shuffle=True, categories=categories, remove=('headers', 'footers', 'quotes'))
  
# You can check the target names (categories) and some data files by following commands.
print (newsgroups_train.target_names) #prints all the categories
print("\n".join(newsgroups_train.data[0].split("\n")[:3])) #prints first line of the first data file
print (newsgroups_train.target_names)
print (len(newsgroups_train.data))
  
texts = []
  
labels=newsgroups_train.target
texts = newsgroups_train.data
 
from nltk.corpus import stopwords
 
import textacy
from textacy.vsm import Vectorizer
import textacy.tm
 
terms_list=[[tok  for tok in doc.split() if tok not in stopwords.words('english') ] for doc in texts]
  
 
count=0           
for doc in terms_list:
 for word in doc:   
   print (word) 
   if word == "|>" or word == "|>" or word == "_" or word == "-" or word == "#":
         terms_list[count].remove (word)
   if word == "=":
         terms_list[count].remove (word)
   if word == ":":
         terms_list[count].remove (word)    
   if word == "_/":
         terms_list[count].remove (word)  
   if word == "I" or word == "A":
         terms_list[count].remove (word)
   if word == "The" or word == "But" or word=="If" or word=="It":
         terms_list[count].remove (word)       
 count=count+1
       
 
print ("=====================terms_list===============================")
print (terms_list)
 
 
vectorizer = Vectorizer(tf_type='linear', apply_idf=True, idf_type='smooth')
doc_term_matrix = vectorizer.fit_transform(terms_list)
 
 
print ("========================doc_term_matrix)=======================")
print (doc_term_matrix)
 
 
 
#initialize and train a topic model:
model = textacy.tm.TopicModel('nmf', n_topics=20)
model.fit(doc_term_matrix)
 
print ("======================model=================")
print (model)
 
doc_topic_matrix = model.transform(doc_term_matrix)
for topic_idx, top_terms in model.top_topic_terms(vectorizer.id_to_term, topics=[0,1]):
          print('topic', topic_idx, ':', '   '.join(top_terms))
 
for i, val in enumerate(model.topic_weights(doc_topic_matrix)):
     print(i, val)
      
      
print   ("doc_term_matrix")     
print   (doc_term_matrix)   
print ("vectorizer.id_to_term")
print (vectorizer.id_to_term)
          
 
model.termite_plot(doc_term_matrix, vectorizer.id_to_term, topics=-1,  n_terms=25, sort_terms_by='seriation')  
plt.show()
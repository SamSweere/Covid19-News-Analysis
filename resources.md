Entry source: https://monkeylearn.com/text-analysis/
Main library: spacy

# Pre-trained models
https://github.com/facebookresearch/InferSent
https://github.com/salesforce/cove
http://nlp.town/blog/off-the-shelf-sentiment-analysis/


# Libraries
word2vec
NLTK
gensim
spacy https://spacy.io/usage/linguistic-features https://spacy.io/usage/spacy-101
StanfordNER
sklearn, dask pipelines: https://medium.com/@chrisfotache/text-classification-in-python-pipelines-nlp-nltk-tf-idf-xgboost-and-more-b83451a327e0


# Preprocessing
1) Tokenization
2) Part-of-speech tagging
3) Syntacitic parsing
    - constituency grammars
    - dependency grammars (often more practical)
4) Lemmatization & Stemming
    - https://stackoverflow.com/questions/15586721/wordnet-lemmatization-and-pos-tagging-in-python
5) Stopword removal
5) Vectorization
    https://monkeylearn.com/blog/beginners-guide-text-vectorization/
    - Bag of words (pre-deep learning era)
      counts, n-grams, tf-idf scores
    - word2vec (Looks best) https://p.migdal.pl/2017/01/06/king-man-woman-queen-why.html
    - Skitp thought vectors; sentence vectorization; https://arxiv.org/abs/1506.06726


# Analysis

Bag of words vectorization
(word2vec?)

## Text Classification

## Topic Analysis
LDA (Latent Dirichlet Allocation
https://towardsdatascience.com/building-a-topic-modeling-pipeline-with-spacy-and-gensim-c5dc03ffc619
https://towardsdatascience.com/topic-modelling-in-python-with-nltk-and-gensim-4ef03213cd21)
https://github.com/FelixChop/MediumArticles/blob/master/LDA-BBC.ipynb
https://cran.r-project.org/web/packages/LDAvis/vignettes/details.pdf


## Classification
https://medium.com/@chrisfotache/text-classification-in-python-pipelines-nlp-nltk-tf-idf-xgboost-and-more-b83451a327e0

## Keyword Extraction

## Named Entity Recognition
https://towardsdatascience.com/named-entity-recognition-with-nltk-and-spacy-8c4a7d88e7da
https://medium.com/sicara/train-ner-model-with-nltk-stanford-tagger-english-french-german-6d90573a9486
## papers
https://arxiv.org/pdf/1812.09449.pdf
https://arxiv.org/pdf/1910.11470.pdf 

## Clustering

## Sentiment Analysis
http://nlp.town/blog/off-the-shelf-sentiment-analysis/

## Text Extraction
Regex
Conditional Random Fields (better)


# Evaluation
Accuracy, Precision, Recall, F1-Score, Cross-Validation, ROUGE metrics


# Visualization

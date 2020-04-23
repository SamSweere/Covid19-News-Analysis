Main source: https://monkeylearn.com/text-analysis/


# Pre-trained models
https://github.com/facebookresearch/InferSent
https://github.com/salesforce/cove
http://nlp.town/blog/off-the-shelf-sentiment-analysis/


# Libraries
word2vec
NLTK
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
https://towardsdatascience.com/topic-modelling-in-python-with-nltk-and-gensim-4ef03213cd21)
https://github.com/FelixChop/MediumArticles/blob/master/LDA-BBC.ipynb
https://cran.r-project.org/web/packages/LDAvis/vignettes/details.pdf


## Classification
https://medium.com/@chrisfotache/text-classification-in-python-pipelines-nlp-nltk-tf-idf-xgboost-and-more-b83451a327e0

## Keyword Extraction

## Named Entity Recognition

## Clustering

## Sentiment Analysis
http://nlp.town/blog/off-the-shelf-sentiment-analysis/

## Text Extraction
Regex
Conditional Random Fields (better)


# Evaluation
Accuracy, Precision, Recall, F1-Score, Cross-Validation, ROUGE metrics


# Visualization

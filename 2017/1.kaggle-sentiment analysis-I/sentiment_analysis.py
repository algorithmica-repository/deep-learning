import os
from sklearn.feature_extraction import text
from sklearn import ensemble 
import pandas as pd
import numpy as np
import re
import nltk
from bs4 import BeautifulSoup
from nltk import corpus

def preprocess_review(review):        #
        # 1. Remove HTML
        review_text = BeautifulSoup(review).get_text()
        #
        # 2. Remove non-letters
        review_text = re.sub("[^a-zA-Z]"," ", review_text)
        #
        # 3. Convert words to lower case
        return review_text.lower()

def stop_words():
    #nltk.download()
    return list(corpus.stopwords.words("english"))
    
def tokenize(review):
    return review.split()

os.chdir("E:/")
    
movie_train = pd.read_csv("labeledTrainData.tsv", header=0, 
                    delimiter="\t", quoting=3)
movie_train.shape
movie_train.info()
movie_train.loc[0:4,'review']

#text cleaning example for one sample review
review_tmp = movie_train['review'][0]
review_tmp = BeautifulSoup(review_tmp).get_text()
review_tmp = re.sub("[^a-zA-Z]"," ", review_tmp)
review_tmp_words = review_tmp.split(' ')

vectorizer = text.CountVectorizer(preprocessor = preprocess_review, \
                                  tokenizer = tokenize,    \
                                  stop_words = 'english',   \
                                  max_features = 5000)

features = vectorizer.fit_transform(movie_train.loc[0:3,'review']).toarray()

vectorizer.get_stop_words()
vectorizer.vocabulary_
vocab = vectorizer.get_feature_names()

dist = np.sum(features, axis=0)

for tag, count in zip(vocab, dist):
    print(count, tag)
    
forest = ensemble.RandomForestClassifier(n_estimators = 100) 

# Fit the forest to the training set, using the bag of words as 
# features and the sentiment labels as the response variable

forest = forest.fit(features, movie_train['sentiment'] )


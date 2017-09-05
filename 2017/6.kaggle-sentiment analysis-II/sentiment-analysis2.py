import os
from sklearn import ensemble 
import pandas as pd
import numpy as np
import re
from bs4 import BeautifulSoup
from nltk import word_tokenize          
from sklearn import model_selection
from nltk.corpus import stopwords

os.chdir('/home/algo/deeplearning')
GLOVE_DIR = '/home/algo/deeplearning/glove.6B' 
GLOVE_FILE = 'glove.6B.50d.txt' 

def preprocess(review):        
        # 1. Remove HTML
        review_text = BeautifulSoup(review).get_text()
        #
        # 2. Remove non-letters
        review_text = re.sub("[^a-zA-Z]"," ", review_text)
        #
        # 3. Convert words to lower case
        return review_text.lower()
        
def tokenize(review):
    return word_tokenize(review)

def removeStopWords(review_words):
    words = [w for w in review_words if not w in stopwords.words("english")]
    return words

def cleanReview(review):
    return removeStopWords(tokenize(preprocess(review)))
 
def getEmbeddingIndex(path, file):
    embeddings_index = {}
    f = open(os.path.join(GLOVE_DIR,file))
    for line in f:
        values = line.split()
        word = values[0]
        value = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = value
    f.close()
    return embeddings_index

def getDocumentVector_Avg(review, embeddings_index):
    documentVec = np.zeros(50,dtype="float32")

    nwords = 0.
    for word in review:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            nwords = nwords + 1.
            documentVec = np.add(documentVec, embedding_vector)

    documentVec = np.divide(documentVec,nwords)
    return documentVec

def getDocumentVectors(reviews, embeddings_index):
    counter = 0
    reviewFeatureVecs = np.zeros( (len(reviews),50), dtype="float32")
    for review in reviews:
        reviewFeatureVecs[counter] = getDocumentVector_Avg(review, embeddings_index)
        counter = counter + 1
    return reviewFeatureVecs

  
movie_train = pd.read_csv("labeledTrainData.tsv", header=0, 
                    delimiter="\t", quoting=3)
movie_train.shape
movie_train.info()
movie_train.loc[0:4,'review']

#preprocess text
clean_train_reviews = movie_train['review'][:2].apply(cleanReview)
len(clean_train_reviews)

#load word embeddings
embeddings_index = getEmbeddingIndex(GLOVE_DIR, GLOVE_FILE)
len(embeddings_index)

#extract document vectors for reviews
X_train = getDocumentVectors(clean_train_reviews, embeddings_index)
y_train = movie_train['sentiment']

#build model with the extracted features
rf_estimator = ensemble.RandomForestClassifier(n_estimators = 100) 
model_selection.cross_val_score(rf_estimator, X_train, y_train, cv = 10).mean()
forest = rf_estimator.fit(X_train, y_train)

movie_test = pd.read_csv("testData.tsv", header=0, 
                    delimiter="\t", quoting=3)
movie_test.shape
movie_train.info()



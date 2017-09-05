import numpy as  np
import pandas as pd
import os
from bs4 import BeautifulSoup
from nltk import word_tokenize          
from nltk.corpus import stopwords
import re
from keras.preprocessing.text import Tokenizer
from keras.layers import Embedding, Dense
from keras.layers.core import Flatten
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential

os.chdir('/home/algo/deeplearning')
GLOVE_DIR = '/home/algo/deeplearning/glove.6B' 
GLOVE_FILE = 'glove.6B.50d.txt'
MAX_SEQ_LEN = 300 
WORD_EMB_SIZE = 50
MAX_VOCAB_SIZE = 2000

def preprocess(review):        
        # 1. Remove HTML
        review = BeautifulSoup(review).get_text()

        # 2. Remove non-letters from review
        review = re.sub("[^a-zA-Z]"," ", review)
        
        # 3. normalize review
        review = review.lower()
        
        # 4. tokenize the words for stopword removal
        tokens = word_tokenize(review)
        
        # 5. remove stop words
        stop_words = set(stopwords.words("english"))
        review = [w for w in tokens if not w in stop_words]

        return ' '.join(review)
        
def tokenize(reviews):
    tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, lower=False, split=' ')
    tokenizer.fit_on_texts(reviews)
    return tokenizer

def getSequences(reviews, tokenizer):
    sequences = tokenizer.texts_to_sequences(reviews)
    sequences = pad_sequences(sequences, maxlen=MAX_SEQ_LEN)
    return sequences

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

def getEmbeddingWeightMatrix(embeddings_index, word_index):    
    embedding_matrix = np.zeros((len(word_index)+1, WORD_EMB_SIZE))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix

movie_train = pd.read_csv("labeledTrainData.tsv", header=0, 
                    delimiter="\t", quoting=3)
movie_train.shape
movie_train.info()
movie_train.loc[:2,'review']

#preprocess text
clean_train_reviews = movie_train['review'][:2].apply(preprocess)
len(clean_train_reviews)

#get the vocabulary across all reviews
tokenizer = tokenize(clean_train_reviews)
tokenizer.word_index
print(len(tokenizer.word_index))

#convert the documents into fixed length sequences
reviews_sequences = getSequences(clean_train_reviews, tokenizer)
len(reviews_sequences[0])
len(reviews_sequences[1])

#load word embeddings from glove pretrained file
embeddings_index = getEmbeddingIndex(GLOVE_DIR, GLOVE_FILE)
len(embeddings_index)

#get embedding layer weight matrix
embedding_weight_matrix = getEmbeddingWeightMatrix(embeddings_index, tokenizer.word_index)
embedding_weight_matrix.shape

X_train = reviews_sequences
y_train = movie_train['sentiment'][:2]

model = Sequential()
model.add( Embedding(input_dim = embedding_weight_matrix.shape[0],
                     output_dim = embedding_weight_matrix.shape[1],
                            weights=[embedding_weight_matrix],
                            input_length=MAX_SEQ_LEN,
                            trainable=False) )
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train, y_train)

model.predict(X_train)





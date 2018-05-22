import os
import pandas as pd
import numpy as np
import re
from bs4 import BeautifulSoup
from nltk import word_tokenize          
from nltk.corpus import stopwords
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers import Input, Bidirectional
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.layers import Dense, Conv1D, MaxPooling1D, Dropout
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint

data_dir = 'C:/Users/Thimma Reddy/Downloads'
glove_file = 'C:/Users/Thimma Reddy/Downloads/glove.6B/glove.6B.50d.txt'
word_embed_size = 50
batch_size = 64
epochs = 1
seq_maxlen = 80

def cleanReview(review):        
        #1.Remove HTML
        review_text = BeautifulSoup(review).get_text()
        #2.Remove non-letters
        review_text = re.sub("[^a-zA-Z]"," ", review_text)
        #3.Convert words to lower case
        review_text = review_text.lower()
        #4.remove stop words
        review_words = word_tokenize(review)
        words = [w for w in review_words if not w in stopwords.words("english")]
        return ' '.join(words)
    
def buildVocabulary(reviews):
    tokenizer = Tokenizer(lower=False, split=' ')
    tokenizer.fit_on_texts(reviews)
    return tokenizer

def getSequences(reviews, tokenizer, seq_maxlen):
    reviews_seq = tokenizer.texts_to_sequences(reviews)
    return np.array(pad_sequences(reviews_seq, maxlen=seq_maxlen))

def loadGloveWordEmbeddings(glove_file):
    embedding_vectors = {}
    f = open(glove_file,encoding='utf8')
    for line in f:
        values = line.split()
        word = values[0]
        value = np.asarray(values[1:], dtype='float32')
        embedding_vectors[word] = value
    f.close()
    return embedding_vectors

def getEmbeddingWeightMatrix(embedding_vectors, word2idx):    
    embedding_matrix = np.zeros((len(word2idx)+1, word_embed_size))
    for word, i in word2idx.items():
        embedding_vector = embedding_vectors.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix

imdb_train = pd.read_csv(os.path.join(data_dir,"labeledTrainData.tsv"), header=0, 
                    delimiter="\t", quoting=3)
imdb_train.shape
imdb_train.info()
imdb_train.loc[0:4,'review']

#preprocess text
review_train_clean = imdb_train['review'][0:4].map(cleanReview)
print(len(review_train_clean))

#build vocabulary over all reviews
tokenizer = buildVocabulary(review_train_clean)
vocab_size = len(tokenizer.word_index) + 1
print(tokenizer.word_index)
print(vocab_size)

X_train = getSequences(review_train_clean, tokenizer, seq_maxlen)
y_train = np_utils.to_categorical(imdb_train['sentiment'][0:4])

#load pre-trained word embeddings
embedding_vectors = loadGloveWordEmbeddings(glove_file)
print(len(embedding_vectors))
#get embedding layer weight matrix
embedding_weight_matrix = getEmbeddingWeightMatrix(embedding_vectors, tokenizer.word_index)
print(embedding_weight_matrix.shape)

#build model        
input = Input(shape=(X_train.shape[1],))

inner = Embedding(input_dim=vocab_size, output_dim=word_embed_size, 
                   input_length=seq_maxlen, weights=[embedding_weight_matrix], 
                   trainable = False) (input)
inner = Conv1D(64, 5, padding='valid', activation='relu', strides=1)(inner)
inner = MaxPooling1D(pool_size=4)(inner)
inner = Bidirectional(LSTM(100, return_sequences=False)(inner))
inner = Dropout(0.3)(inner)
inner = Dense(50, activation='relu')(inner)
output = Dense(2, activation='softmax')(inner)

model = Model(inputs = input, outputs = output)
model.compile(Adam(lr=0.01), 'categorical_crossentropy', metrics=['accuracy'])

save_weights = ModelCheckpoint('model.h5', monitor='val_loss', save_best_only=True)
history = model.fit(X_train, y_train, verbose=1, epochs=epochs, batch_size=batch_size, 
                    callbacks=[save_weights], validation_split=0.1)

imdb_test = pd.read_csv(os.path.join(data_dir,"testData.tsv"), header=0, 
                    delimiter="\t", quoting=3)
imdb_test.shape
imdb_test.info()
imdb_test.loc[0:4,'review']

#preprocess text
review_test_clean = imdb_test['review'][0:4].map(cleanReview)
print(len(review_test_clean))

X_test = getSequences(review_test_clean, tokenizer, seq_maxlen)
imdb_test['sentiment'] = model.predict(X_test).argmax(axis=-1)
imdb_test.to_csv(os.path.join(data_dir,'submission.csv'), columns=['id','sentiment'], index=False)
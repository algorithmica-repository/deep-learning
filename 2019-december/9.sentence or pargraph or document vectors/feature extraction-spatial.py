import sys
sys.path.append("G:/New Folder/utils")
import os
import re
import numpy as np
import pandas as pd
import seaborn as sns
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.layers.embeddings import Embedding
from keras.layers import Input, LSTM
import keras_utils as kutils

def cleanText(reviews):
    text = [re.sub(r'([^\s\w]|_)+', '', sentence) for sentence in reviews]
    text = [sentence.lower().split() for sentence in text]
    return text
    
def buildVocabulary(reviews):
    tokenizer = Tokenizer(lower=False, split=' ')
    tokenizer.fit_on_texts(reviews)
    return tokenizer

def getSequences(reviews, tokenizer, seq_maxlen):
    reviews_seq = tokenizer.texts_to_sequences(reviews)
    return np.array(pad_sequences(reviews_seq, maxlen=seq_maxlen))

def loadWordVectors(glove_file):
    embedding_vectors = {}
    f = open(glove_file,encoding='utf8')
    for line in f:
        values = line.split()
        word = values[0]
        value = np.asarray(values[1:], dtype='float32')
        embedding_vectors[word] = value
    f.close()
    return embedding_vectors

def getEmbeddingLayerWeightMatrix(embedding_vectors, word2idx):    
    embedding_matrix = np.zeros((len(word2idx)+1, word_embed_size))
    for word, i in word2idx.items():
        embedding_vector = embedding_vectors.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix

data = ['I love machine learning',
        'I like reading books',
        'Python is beautiful',
        'R is horrible.',
        'Machine learning is cool!',
        'I really like NLP',
        'I really like deep learning']

#clean text
text = cleanText(data)

#build vocabulary over all sentences
tokenizer = buildVocabulary(text)
vocab_size = len(tokenizer.word_index) + 1
print(tokenizer.word_index)
print(vocab_size)

word_embed_size = 50
seq_maxlen = 6
glove_file = 'G:/glove.6B.50d.txt'

#get fixed length sequences of sentences
X_train = getSequences(text, tokenizer, seq_maxlen)

#get word vectors
word_vectors = loadWordVectors(glove_file)
#print(word_vectors)
#get embedding layer weight matrix
emb_layer_weights = getEmbeddingLayerWeightMatrix(word_vectors, tokenizer.word_index)
print(emb_layer_weights)

#build model        
input = Input(shape=(X_train.shape[1],))

inner = Embedding(input_dim=vocab_size, output_dim=word_embed_size, 
                   input_length=seq_maxlen, weights=[emb_layer_weights], 
                   trainable = False) (input)
lstm1 = LSTM(200, return_sequences=True)(inner)
features = LSTM(100, return_sequences=False)(lstm1)

model = Model(inputs = input, outputs = features)
print(model.summary())
feature_vectors = model.predict(X_train)
print(feature_vectors)

#visualize sentence vectors
kutils.viz_vectors(feature_vectors, data)
kutils.viz_vectors_corr(feature_vectors, data)
kutils.viz_vectors_lower_dim(feature_vectors, data)

#visualize layer activations
act = kutils.get_activations(model, X_train[0:1])  # with just one sample.
kutils.display_activations(act, directory=os.path.join("G:/", 'digit_activations'), save=True)
kutils.display_heatmaps(act,tmp[0:1], directory=os.path.join("G:/", 'digit_heatmaps'), save=True)
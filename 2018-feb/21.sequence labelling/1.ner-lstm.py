from keras.layers import Dense, Dropout, Input, TimeDistributed
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.callbacks import ModelCheckpoint
import numpy as np
import os
from keras.utils import np_utils
from sklearn.metrics import classification_report

data_dir = 'E:/ner/data'
glove_file = 'C:/Users/Thimma Reddy/Downloads/glove.6B/glove.6B.50d.txt'
word_embed_size = 50
batch_size = 64
epochs = 2

def buildVocabulary(words):
    tokenizer = Tokenizer(lower=True, split=' ', filters='')
    tokenizer.fit_on_texts(words)
    return tokenizer

def getSequences(sentences, tokenizer, seq_maxlen, pad_value):
    sequence = tokenizer.texts_to_sequences(sentences)
    return pad_sequences(sequence, maxlen=seq_maxlen, padding='post', value=pad_value)

def load_data(filename):
    f = open(filename, encoding="latin1")
    sentences = []
    tags = []
    sentence = []
    tag = []
    for line in f:
        if len(line)==0 or line.startswith('-DOCSTART') or line[0]=="\n":
            if len(sentence) > 0:
                sentences.append(' '.join(sentence))
                tags.append(' '.join(tag))
                sentence = []
                tag=[]
            continue
        splits = line.split(' ')
        sentence.append(splits[0].lower())
        tag.append(splits[-1].rstrip('\n').lower())
    f.close()
    return sentences, tags

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

def decode_output(X_test, y_pred, y_test, tokenizer_words, tokenizer_tags, seq_maxlen):
    y_pred_tags = []
    y_test_tags = []
    word_index_reverse = {}
    for k,v in tokenizer_words.word_index.items():
        word_index_reverse[v] = k
    
    tag_index_reverse = {}
    for k,v in tokenizer_tags.word_index.items():
        tag_index_reverse[v] = k
  
    for i in range(X_test.shape[0]):    
        inp = X_test[i]
        out = y_test[i]
        pred = np.argmax(y_pred[i], axis=-1)
        test_str = []
        test_pred = []
        test_out = []
        for j in range(1,seq_maxlen):
            if inp[j] != 0:
                test_str.append(word_index_reverse[inp[j]]) 
                test_pred.append(tag_index_reverse[pred[j]])
                test_out.append(tag_index_reverse[out[j]])
                y_pred_tags.append(tag_index_reverse[pred[j]])
                y_test_tags.append(tag_index_reverse[out[j]])

        print(' '.join(test_str))
        print(' '.join(test_out))
        print(' '.join(test_pred))
    return y_pred_tags, y_test_tags

#load the train data from train file
sentences_train, tags_train = load_data(os.path.join(data_dir, 'train.txt'))               

#build vocabulary
tokenizer_words = buildVocabulary(sentences_train)
print(tokenizer_words.word_index)
tokenizer_tags = buildVocabulary(tags_train)
print(tokenizer_tags.word_index)
vocab_size = len(tokenizer_words.word_index) + 1
print(vocab_size)
num_tags = len(tokenizer_tags.word_index) + 1
print(num_tags)

seq_maxlen = 52
X_train = getSequences(sentences_train, tokenizer_words, seq_maxlen, 0)
y_train = getSequences(tags_train, tokenizer_tags, seq_maxlen, tokenizer_tags.word_index['o'])
y_train = np.array([np_utils.to_categorical(i, num_classes=num_tags) for i in y_train])

sentences_val, tags_val = load_data(os.path.join(data_dir, 'validation.txt'))               
X_validation = getSequences(sentences_val, tokenizer_words, seq_maxlen, 0)
y_validation = getSequences(tags_val, tokenizer_tags, seq_maxlen, tokenizer_tags.word_index['o'])
y_validation = np.array([np_utils.to_categorical(i, num_classes=num_tags) for i in y_validation])

#get embedding layer weight matrix
embedding_vectors = loadGloveWordEmbeddings(glove_file)
print(len(embedding_vectors))
embedding_weight_matrix = getEmbeddingWeightMatrix(embedding_vectors, tokenizer_words.word_index)
print(embedding_weight_matrix.shape)

#build model        
input = Input(shape=(X_train.shape[1],))

model = Embedding(input_dim=vocab_size, output_dim=word_embed_size, 
                   input_length=seq_maxlen, weights=[embedding_weight_matrix], 
                   trainable = False) (input)
model = LSTM(100, return_sequences=True)(model)
output = TimeDistributed(Dense(num_tags, activation="softmax"))(model)  

model = Model(inputs=input, outputs=output)
print(model.summary())

model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])

save_weights = ModelCheckpoint('model.h5', monitor='val_loss', save_best_only=True)

model.fit(x=X_train, y=y_train, batch_size=batch_size,
          epochs=epochs, validation_data=(X_validation, y_validation), callbacks=[save_weights])

sentences_test, tags_test = load_data(os.path.join(data_dir, 'test.txt'))               
X_test = getSequences(sentences_test, tokenizer_words, seq_maxlen, 0)
y_test = getSequences(tags_test, tokenizer_tags, seq_maxlen, tokenizer_tags.word_index['o'])

y_pred = model.predict(X_test)

y_pred_tags, y_test_tags = decode_output(X_test, y_pred, y_test, tokenizer_words, tokenizer_tags, seq_maxlen)

report = classification_report(y_pred=y_pred_tags, y_true=y_test_tags)
print(report)

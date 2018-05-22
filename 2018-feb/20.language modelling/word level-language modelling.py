from keras.models import Model
from keras.layers import Dense, Embedding,Dropout
from keras.layers import LSTM, TimeDistributed, Input
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras.optimizers import RMSprop
import re


maxlen = 40
step = 1
batch_size = 128
epochs = 20
word_embed_size = 50
file = 'E:/language modelling/sample.txt'
glove_file = 'C:/Users/Algorithmica/Downloads/glove.6B/glove.6B.50d.txt'

raw_text = open(file,'r').read()
raw_text = [line.strip() for line in raw_text.split('\n')]
raw_text = ' '.join(raw_text)
clean_text = re.sub("[^a-zA-Z]"," ", raw_text)
clean_text = clean_text.lower()
words = clean_text.split()

input_sentences = []
output_sentences = []
for i in range(0, len(words) - maxlen + 1, step):
    input_sentences.append(' '.join(words[i: i + maxlen]))
    output_sentences.append(' '.join(words[i+1: i + maxlen + 1]))
print('nb sequences:', len(input_sentences))
print('nb sequences:', len(output_sentences))

input_sentences = input_sentences[:-2]
output_sentences = output_sentences[:-2]

tokenizer = Tokenizer(lower=True, split=' ')
tokenizer.fit_on_texts(words)
print(tokenizer.word_counts)
print(tokenizer.word_index)
vocab_size = len(tokenizer.word_index)+1
print(vocab_size)

input_sequences = tokenizer.texts_to_sequences(input_sentences)
X_train = np.array(input_sequences)
output_sequences = tokenizer.texts_to_sequences(output_sentences)
y_train = np_utils.to_categorical(output_sequences, vocab_size)

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

#load pre-trained word embeddings
embedding_vectors = loadGloveWordEmbeddings(glove_file)
print(len(embedding_vectors))
#get embedding layer weight matrix
embedding_weight_matrix = getEmbeddingWeightMatrix(embedding_vectors, tokenizer.word_index)
print(embedding_weight_matrix.shape)

input_size = Input(shape=(X_train.shape[1:]),)
inner = Embedding(input_dim=vocab_size, output_dim=word_embed_size, 
                   input_length=maxlen,
                   weights=[embedding_weight_matrix], trainable=False)(input_size)
inner = LSTM(512, return_sequences=True)(inner)
inner = Dropout(0.2)(inner)
inner = LSTM(512, return_sequences=True)(inner)
inner = Dropout(0.2)(inner)
output = TimeDistributed(Dense(vocab_size, activation='softmax'))(inner)
model = Model(inputs = input_size, outputs = output)
print(model.summary())

optimizer = RMSprop(lr=0.01)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics =['accuracy'])

save_weights = ModelCheckpoint('char_model.h5', monitor='val_loss', save_best_only=True)

model.fit(X_train, y_train, validation_split=0.05, 
                    batch_size=batch_size, epochs=epochs, shuffle=True,
                    callbacks=[save_weights])

word_index_reverse = {}
for k,v in tokenizer.word_index.items():
    word_index_reverse[v] = k
 
seed_text = 'supposing that Truth is a woman what then? Is there not ground'
splits = seed_text.split(' ')
if len(splits) < maxlen:
        padding = ' '*(maxlen - len(splits))
        seed_text = padding + ' ' + seed_text
else:
        seed_text = seed_text[len(splits) - maxlen:]
test_sentence = seed_text
print('Seed sentence: "' + test_sentence + '"')
generated = ' '
for i in range(20):
    test_sequences = tokenizer.texts_to_sequences([test_sentence])
    X_test = np.array(test_sequences)
    preds = model.predict(X_test, verbose=0)[0][-1]
    next_word = word_index_reverse[np.argmax(preds)]
    
    generated += ' ' + next_word
    test_sentence = test_sentence[1:] + ' ' + next_word

print(generated)
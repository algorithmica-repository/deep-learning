from keras.models import Sequential
from keras.layers import Dense, Embedding,Dropout
from keras.layers import LSTM
import numpy as np
import utils
from keras.preprocessing.text import Tokenizer
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import RMSprop
import re
from keras.preprocessing.sequence import pad_sequences

maxlen = 40
step = 1
batch_size = 128
epochs = 20
word_embed_size = 50
file = 'E:/text-prediction/nietzsche.txt'
glove_file = 'C:/Users/Thimma Reddy/Downloads/glove.6B/glove.6B.50d.txt'

raw_text = open(file,'r').read()
raw_text = [line.strip() for line in raw_text.split('\n')]
raw_text = ' '.join(raw_text)
clean_text = re.sub("[^a-zA-Z]"," ", raw_text)
clean_text = clean_text.lower()
words = clean_text.split()

text_sequences = []
next_word = []
for i in range(0, len(words) - maxlen, step):
    text_sequences.append(' '.join(words[i: i + maxlen]))
    next_word.append(words[i + maxlen])
print('nb sequences:', len(text_sequences))

tokenizer = Tokenizer(malower=True, split=' ')
tokenizer.fit_on_texts(words)
print(tokenizer.word_counts)
print(tokenizer.word_index)
vocab_size = len(tokenizer.word_index)+1

train_sequences = tokenizer.texts_to_sequences(text_sequences)
X_train = np.array(train_sequences)
target = tokenizer.texts_to_sequences(next_word)
y_train = np_utils.to_categorical(target, vocab_size)

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

model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=word_embed_size, 
                   input_length=maxlen,
                   weights=[embedding_weight_matrix], trainable=False))
model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dense(vocab_size, activation='softmax'))
print(model.summary())

optimizer = RMSprop(lr=0.01)
model.compile(optimizer=optimizer, loss='categorical_crossentropy')

save_weights = ModelCheckpoint('char_model.h5', monitor='val_loss', save_best_only=True)

history = model.fit(X_train, y_train, validation_split=0.05, 
                    batch_size=batch_size, epochs=epochs, shuffle=True,
                    callbacks=[save_weights])
utils.plot_loss_accuracy(history)

word_index_reverse = {}
for k,v in tokenizer.word_index.items():
    word_index_reverse[v] = k
 
start = np.random.randint(0, len(train_sequences)-1)
test_sequence = train_sequences[start]
generated = []
print("Seed sentence: ", ' '.join(word_index_reverse[index] for index in test_sequence) )

for i in range(20):
    print(test_sequence)
    X_test = np.expand_dims(np.array(test_sequence), axis=0)
    preds = model.predict(X_test, verbose=0)[0]
    next_index = np.argmax(preds)
    next_word = word_index_reverse[next_index]
    generated.append(next_word)
    test_sequence.append(next_index)
    test_sequence = test_sequence[1:]
    print(next_word)
print("Generated Sequence: ", ' '.join(generated))
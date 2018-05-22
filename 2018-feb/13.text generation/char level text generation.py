from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
import numpy as np
import utils
from keras.preprocessing.text import Tokenizer
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import RMSprop

maxlen = 40
step = 3
batch_size = 128
epochs = 20
path = 'E:/text-prediction/nietzsche.txt'

text = open(path).read()
print('corpus length:', len(text))

sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('nb sequences:', len(sentences))

tokenizer = Tokenizer(lower=True, char_level=True, split=' ')
tokenizer.fit_on_texts(text)
print(tokenizer.word_counts)
print(tokenizer.word_index)
vocab_size = len(tokenizer.word_index)+1

train_sequences = tokenizer.texts_to_sequences(sentences)
X_train = np_utils.to_categorical(train_sequences, vocab_size)
target = tokenizer.texts_to_sequences(next_chars)
y_train = np_utils.to_categorical(target, vocab_size)

model = Sequential()
model.add(LSTM(128, input_shape=(maxlen, vocab_size)))
model.add(Dense(128, activation='relu'))
model.add(Dense(vocab_size, activation='softmax'))
print(model.summary())

optimizer = RMSprop(lr=0.01)
model.compile(optimizer=optimizer, loss='categorical_crossentropy')

early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto')   
save_weights = ModelCheckpoint('char_model.h5', monitor='val_loss', save_best_only=True)

history = model.fit(X_train, y_train, validation_split=0.05, 
                    batch_size=batch_size, epochs=epochs, shuffle=True,
                    callbacks=[save_weights, early_stopping])
utils.plot_loss_accuracy(history)

word_index_reverse = {}
for k,v in tokenizer.word_index.items():
    word_index_reverse[v] = k
    
seed_text = 'have failed to understand women'
if len(seed_text) < maxlen:
        padding = ' '*(maxlen - len(seed_text))
        seed_text = padding + seed_text
else:
        seed_text = seed_text[len(seed_text) - maxlen:]
test_sentence = seed_text
print('Seed sentence: "' + test_sentence + '"')

generated = ''
for i in range(200):
    test_sequences = tokenizer.texts_to_sequences(test_sentence)
    X_test = np_utils.to_categorical(test_sequences, vocab_size)
    X_test = np.expand_dims(X_test, axis=0)

    preds = model.predict(X_test, verbose=0)[0]
    next_char = word_index_reverse[np.argmax(preds)]
    
    generated += next_char
    test_sentence = test_sentence[1:] + next_char
    print(next_char)

print(generated)

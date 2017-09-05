from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

#feeding text to tokenizer
tokenizer = Tokenizer()
texts = ["The sun is shining in June!","September is grey.","Life is beautiful in August.","I like it","This and other things?"]
tokenizer.fit_on_texts(texts)

#using tokenizer
vocab_size = len(tokenizer.word_index) + 1
print(vocab_size)
print(tokenizer.word_index)
print(tokenizer.word_counts)
print(tokenizer.document_count)

MAX_SEQUENCE_LENGTH = 12
sequences = tokenizer.texts_to_sequences(["June is beautiful and I like it!"])
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)


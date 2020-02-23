from gensim import models
import numpy as np
import pandas as pd
import keras_utils as kutils
   
embeddings = {}
fasttext_file = 'C:/Users/Algorithmica/Downloads/wiki-news-300d-1M.vec/wiki-news-300d-1M.txt'

f = open(fasttext_file,  'r', encoding='utf8')

lines = f.readlines()[1:]
f.close()
for line in lines:
    values = line.split()
    word = values[0]
    value = np.asarray(values[1:], dtype='float32')
    embeddings[word] = value
 
print('Loaded %s word vectors.' % len(embeddings))
print(embeddings['man'])
print(embeddings['woman'])
print(embeddings['guy'])
print(embeddings['boy'])

word_vectors = list(embeddings.values())[100:150]
labels = list(embeddings.keys())[100:150]
kutils.viz_vectors(word_vectors, labels)
kutils.viz_vectors_corr(word_vectors, labels)
kutils.viz_vectors_lower_dim(word_vectors, labels)

word_vectors = [embeddings['man'], embeddings['woman'], embeddings['boy'], embeddings['guy'], embeddings['cat'], embeddings['dog']]
labels = ['man', 'woman', 'boy', 'guy', 'cat', 'dog']
kutils.viz_vectors(word_vectors, labels)
kutils.viz_vectors_corr(word_vectors, labels)
kutils.viz_vectors_lower_dim(word_vectors, labels)
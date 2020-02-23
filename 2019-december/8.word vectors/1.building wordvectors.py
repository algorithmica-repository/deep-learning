from gensim import models
import numpy as np
import pandas as pd
import keras_utils as kutils  
import re

data = ['I love machine learning',
        'I like reading books',
        'Python is beautiful',
        'R is horrible.',
        'Machine learning is cool!',
        'I really like NLP']

# pre-process our text
text = [re.sub(r'([^\s\w]|_)+', '', sentence) for sentence in data]
text = [sentence.lower().split() for sentence in text]

# train Word2Vec model on our data
word_model = models.Word2Vec(text, size=50, min_count=1, iter=100)
print(word_model.wv.vectors)
print(word_model.wv.vocab)
print(len(word_model.wv.vocab))
print(word_model.wv['like'])
word_model.wv.most_similar('python')

# train Fast model on our data
word_model = models.FastText(text, size=50, min_count=1, iter=100)
print(word_model.wv.vectors)
print(word_model.wv.vocab)
print(word_model.wv['like'])
word_model.wv.most_similar('python')

word_vectors = list(word_model.wv.vectors)
labels = list(word_model.wv.vocab)
kutils.viz_vectors(word_vectors, labels)
kutils.viz_vectors_corr(word_vectors, labels)
kutils.viz_vectors_lower_dim(word_vectors, labels)

import numpy as  np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
    plt.figure(figsize=(18, 18))  # in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label,
                 xy=(x, y),
                 xytext=(5, 2),
                 textcoords='offset points',
                 ha='right',
                 va='bottom')
    plt.savefig(filename)
    
embeddings_index = {}
glove_file = 'E:/glove.6B.50d.txt'

f = open(glove_file, encoding='utf8')
for line in f:
    values = line.split()
    word = values[0]
    value = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = value
f.close()
 
print('Loaded %s word vectors.' % len(embeddings_index))
print(embeddings_index['man'])
print(embeddings_index['woman'])
print(embeddings_index['guy'])
print(embeddings_index['boy'])

print(np.corrcoef(embeddings_index['man'], embeddings_index['woman']))
print(np.corrcoef(embeddings_index['man'], embeddings_index['boy']))
print(np.corrcoef(embeddings_index['man'], embeddings_index['guy']))
print(np.corrcoef(embeddings_index['cat'], embeddings_index['dog']))
print(np.corrcoef(embeddings_index['dog'], embeddings_index['man']))

data = np.array([embeddings_index[key] for key in embeddings_index.keys()])
data.shape
data = data[:100,]
labels = list(embeddings_index.keys())[:100]

tsne = TSNE(perplexity=30.0, n_components=2, n_iter=5000)
low_dim_embedding = tsne.fit_transform(data)
plot_with_labels(low_dim_embedding, labels)

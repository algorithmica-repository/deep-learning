import sys
sys.path.append("E:/New Folder/utils")
import pandas as pd
import numpy as np
import os
from sklearn import decomposition, feature_selection, neighbors, model_selection, manifold, svm
import seaborn as sns
import classification_utils as cutils


path = "C:/Users/Algorithmica/Downloads/digit-recognizer"

digit_train = pd.read_csv(os.path.join(path, "train.csv"))
print(digit_train.shape)
print(digit_train.info())

X = digit_train.iloc[:,1:]/255.0
y = digit_train['label']

X_train, X_eval, y_train, y_eval = model_selection.train_test_split(X, y, test_size=0.1, random_state=1)


zv = feature_selection.VarianceThreshold(threshold=0.0)
X_train1 = zv.fit_transform(X_train)

sns.heatmap(X_train.corr())

lpca = decomposition.PCA(n_components=0.95)
X_train2 = lpca.fit_transform(X_train1)
np.cumsum(lpca.explained_variance_ratio_)

tsne = manifold.TSNE()
tsne_data = tsne.fit_transform(X_train2)
cutils.plot_data_2d_classification(tsne_data, y_train)

kernel_svm_estimator = svm.SVC(kernel='rbf')
kernel_svm_grid = {'gamma':[0.01, 0.1, 1, 2, 5, 10], 'C':[0.001, 0.01, 0.1, 0.5] }
kernel_svm_grid_estimator = model_selection.GridSearchCV(kernel_svm_estimator, kernel_svm_grid, scoring='accuracy', cv=10)
kernel_svm_grid_estimator.fit(X_train2, y_train)
print(kernel_svm_grid_estimator.best_params_)
print(kernel_svm_grid_estimator.best_score_)
print(kernel_svm_grid_estimator.score(X_train2, y_train))

print(kernel_svm_grid_estimator.score(X_eval, y_eval))
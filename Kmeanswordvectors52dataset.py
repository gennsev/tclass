#!/usr/bin/env python
# coding: utf-8

# ## Document Classification by K-Means

# ### Includes

# In[1]:


import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.cluster import KMeans, DBSCAN
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from mpl_toolkits import mplot3d


# ### Vectorizer
# Using GloVe word vectors

# In[2]:


class GloveVectorizer:
  def __init__(self):
    # load in pre-trained word vectors
    print('Loading word vectors...')
    word2vec = {}
    embedding = []
    idx2word = []
    with open('glove.6B.50d.txt',encoding='utf-8', errors='ignore') as f:
      # is just a space-separated text file in the format:
      # word vec[0] vec[1] vec[2] ...
      for line in f:
        values = line.split()
        word = values[0]
        vec = np.asarray(values[1:], dtype='float32')
        word2vec[word] = vec
        embedding.append(vec)
        idx2word.append(word)
    print('Found %s word vectors.' % len(word2vec))

    # save for later
    self.word2vec = word2vec
    self.embedding = np.array(embedding)
    self.word2idx = {v:k for k,v in enumerate(idx2word)}
    self.V, self.D = self.embedding.shape

  def fit(self, data):
    pass

  def transform(self, data):
    X = np.zeros((len(data), self.D))
    n = 0
    emptycount = 0
    for sentence in data:
      tokens = sentence.lower().split()
      vecs = []
      for word in tokens:
        if word in self.word2vec:
          vec = self.word2vec[word]
          vecs.append(vec)
      if len(vecs) > 0:
        vecs = np.array(vecs)
        X[n] = vecs.mean(axis=0)
      else:
        emptycount += 1
      n += 1
    print("Numer of samples with no words found: %s / %s" % (emptycount, len(data)))
    return X

  def fit_transform(self, data):
    self.fit(data)
    return self.transform(data)


# In[3]:


def main():
    train = pd.read_csv('r52-train-all-terms.txt', header=None, sep='\t')
    test = pd.read_csv('r52-test-all-terms.txt', header=None, sep='\t')
    train.columns = ['label', 'content']
    test.columns = ['label', 'content']

    vectorizer = GloveVectorizer()
    Xtrain = vectorizer.fit_transform(train.content)
    Ytrain = train.label

    Xtest = vectorizer.transform(test.content)
    Ytest = test.label

    # create the model, train it, print scores
    model = RandomForestClassifier(n_estimators=200)
    model.fit(Xtrain, Ytrain)
    print("train score:", model.score(Xtrain, Ytrain))
    print("test score:", model.score(Xtest, Ytest))

if __name__ == "__main__":
    main()


# # ----------------------------------------------------------------------------------------------------------

# # Defining new Columns

# In[4]:


#-------------------------------------------------------------------------------#
train = pd.read_csv('r52-train-all-terms.txt', header=None, sep='\t')
test = pd.read_csv('r52-test-all-terms.txt', header=None, sep='\t')
train.columns = ['label', 'content']
test.columns = ['label', 'content']
train['lenght'] = train['content'].str.len()
lenght_mn=train['lenght'].mean()
train['lenght_mean']=(lambda x: train['lenght']/lenght_mn)(train['lenght'].values)
train['words_num'] = train['content'].str.split().str.len()
train['words_len_med'] = train['content'].str.len()/train['words_num']
train['words_num_norm'] = (train['words_num'] - train['words_num'].min())/(train['words_num'].max()-train['words_num'].min())
train['words_len_med_norm'] = (train['words_len_med'] - train['words_len_med'].min())/(train['words_len_med'].max()-train['words_len_med'].min())
train['lenght_norm'] = (train['lenght'] - train['lenght'].min())/(train['lenght'].max()-train['lenght'].min())
train['words_num_norm'] = (train['words_num'] - train['words_num'].min())/(train['words_num'].max()-train['words_num'].min())
train['words_len_med_norm'] = (train['words_len_med'] - train['words_len_med'].min())/(train['words_len_med'].max()-train['words_len_med'].min())

#-------------------------------------------------------------------------------#

train.head()


# # Vectorizer

# In[5]:


vectorizer = GloveVectorizer()
Xtrain = vectorizer.fit_transform(train.content)
model_1=KMeans(n_clusters=8, init='random').fit(Xtrain)
centroids = model_1.cluster_centers_


# # K Means Graph- 2 Features (Normalized)

# In[6]:


zipped_data = np.array(list(zip(train.words_len_med_norm, train.lenght_norm)))
model_2=KMeans(n_clusters=8, init='random').fit(zipped_data)
plt.figure(figsize=(8, 6))
plt.scatter(zipped_data[:,0], zipped_data[:,1], c=model_2.labels_.astype(float))
plt.show()


# # K Means Graph- 2 Features (Non Normalized)

# In[7]:


zipped_data = np.array(list(zip(train.words_len_med, train.lenght)))
model_3=KMeans(n_clusters=8, init='random').fit(zipped_data)
plt.figure(figsize=(8, 6))
plt.scatter(zipped_data[:,0], zipped_data[:,1], c=model_3.labels_.astype(float))
plt.show()


# # Testing Dataset's Novel Features with K-Means

# In[8]:


zipped_data = np.array(list(zip(train.lenght_norm,train.words_num_norm, train.words_len_med_norm)))

model_4_fit=KMeans(n_clusters=8, init='random').fit(zipped_data)


plt.figure(figsize=(8, 6))
plt.scatter(zipped_data[:,0],zipped_data[:,1], zipped_data[:,2], c=model_4_fit.labels_.astype(float))
plt.show()


# # 3D Plotting

# In[9]:


plt.figure(figsize=(8, 6))
ax = plt.axes(projection='3d')
ax.scatter3D(zipped_data[:,0],zipped_data[:,1], zipped_data[:,2], c=model_4_fit.labels_.astype(float))
ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
ax.set_xlabel('lenght_norm')
ax.set_ylabel('words_num_norm')
ax.set_zlabel('words_len_med_norm')
ax.set_title('K-Means Normalized')


# # Elbow Test

# In[10]:


sse = []
list_k = list(range(1, 10))
zipped_data = np.array(list(zip(train.lenght_norm,train.words_num_norm, train.words_len_med_norm)))
Xtrain = vectorizer.fit_transform(train.content)
for k in list_k:
    km = KMeans(n_clusters=k)
    km.fit(zipped_data)
    sse.append(km.inertia_)
# Plot sse against k
plt.figure(figsize=(6, 6))
plt.plot(list_k, sse, '-o')
plt.xlabel(r'Number of clusters *k*')
plt.ylabel('Sum of squared distance');


# In[11]:


sse = []
list_k = list(range(1, 10))
vectorizer = GloveVectorizer()
Xtrain = vectorizer.fit_transform(train.content)
for k in list_k:
    km = KMeans(n_clusters=k)
    km.fit(Xtrain)
    sse.append(km.inertia_)
# Plot sse against k
plt.figure(figsize=(6, 6))
plt.plot(list_k, sse, '-o')
plt.xlabel(r'Number of clusters *k*')
plt.ylabel('Sum of squared distance');


# In[ ]:





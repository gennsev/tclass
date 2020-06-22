
from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future


import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import word2vecVectorizer
import glovevectorizer

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier


def main():
    train = pd.read_csv('train/r8-train-all-terms.txt', header=None, sep='\t')
    test = pd.read_csv('test/r8-test-all-terms.txt', header=None, sep='\t')
    train.columns = ['label', 'content']
    test.columns = ['label', 'content']

    vectorizer = glovevectorizer.GloveVectorizer()
    #vectorizer = word2vecVectorizer.Word2VecVectorizer()
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
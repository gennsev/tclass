import sys
import os
import numpy as np
import pandas as pd
import time
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.cluster import KMeans, DBSCAN


class Evaluator:
    def bench_k_means(self, estimator, name, data, sample_size):
        """
        Different evaluation methods for K-means clustering algorithm
        """
        t0 = time.time()
        labels = data.labels
        estimator.fit(data)
        print('%-9s\t%.2fs\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
            % (name, (time.time() - t0), estimator.inertia_,
                metrics.homogeneity_score(labels, estimator.labels_),
                metrics.completeness_score(labels, estimator.labels_),
                metrics.v_measure_score(labels, estimator.labels_),
                metrics.adjusted_rand_score(labels, estimator.labels_),
                metrics.adjusted_mutual_info_score(labels,  estimator.labels_),
                metrics.silhouette_score(data, estimator.labels_,
                                        metric='euclidean', sample_size=sample_size)))
    def bench_XGB(self, y_true, prediction):
        """ 
        Uses Accuracy Score from Scikit to provide this metric to the model. Prediction must be a DMatrix
        """
        print(accuracy_score(y_true, prediction, normalize=True, sample_weight=None))
    
    
    def bench_randomforest (self,  Ytest, prediction):
        """ 
        Uses Accuracy Score from Scikit to provide this metric to the model. Ytest shall contain the right labels"
        """  
        print(
            accuracy_score(y_true, prediction, normalize=True, sample_weight=None))

    def elbow_test (self, data, max_int):
        """
        Data may include an array of the features that will be involved in the test. The range is varys between 1 and max_int (must be an integer > 1),
        that shall be provided by the user. In cluster analysis, the elbow method is a heuristic used in determining the number of clusters in a data set.
        The method consists of plotting the explained variation as a function of the number of clusters, and picking the elbow of the curve as the number of clusters to use. 
        The same method can be used to choose the number of parameters in other data-driven models, such as the number of principal components to describe a data set.
        """
        sse = []
        list_k = list(range(1, max_int))
        for k in list_k:
            km = KMeans(n_clusters=k)
            km.fit(data)
            sse.append(km.inertia_)
        # Plot sse against k
        plt.figure(figsize=(6, 6))
        plt.plot(list_k, sse, '-o')
        plt.xlabel(r'Number of clusters *k*')
        plt.ylabel('Sum of squared distance')
        

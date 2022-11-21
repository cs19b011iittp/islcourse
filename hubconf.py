import torch
from torch import nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
%matplotlib inline


def get_data_blobs(n_points=100):
  pass
  # write your code here
  # Refer to sklearn data sets
  # write your code ...
  from sklearn.datasets.samples_generator import make_blobs
  X, y = make_blobs(n_samples=300, centers=4,
                       cluster_std=0.60, random_state=0)
  plt.scatter(X[:, 0], X[:, 1], s=50);
  return X,y


def get_data_circles(n_points=100):
  pass
  # write your code here
  # Refer to sklearn data sets
  import sklearn.cluster as skl_cluster
  import sklearn.datasets as skl_data

  circles, circles_clusters = skl_data.make_circles(n_samples=400, noise=.01, random_state=0)

  # cluster with kmeans
  Kmean = skl_cluster.KMeans(n_clusters=2)
  Kmean.fit(circles)
  clusters = Kmean.predict(circles)

  # plot the data, colouring it by cluster
  plt.scatter(circles[:, 0], circles[:, 1], s=15, linewidth=0.1, c=clusters,cmap='flag')
  plt.show()

  # cluster with spectral clustering
  model = skl_cluster.SpectralClustering(n_clusters=2, affinity='nearest_neighbors', assign_labels='kmeans')
  labels = model.fit_predict(circles)
  plt.scatter(circles[:, 0], circles[:, 1], s=15, linewidth=0, c=labels, cmap='flag')
  plt.show()
  return circles, circles_clusters


def get_data_mnist():
  pass
  # write your code here
  # Refer to sklearn data sets
  from keras.datasets import mnist

  (x_train, y_train), (x_test, y_test) = mnist.load_data()
  X = (x_train, x_test)
  y = (y_train, y_test)
  print("Training Data: {}".format(x_train.shape))
  print("Training Labels: {}".format(y_train.shape))
  print("Testing Data: {}".format(x_test.shape))
  print("Testing Labels: {}".format(y_test.shape))
  return X,y


def build_kmeans(X=None,k=10):
  pass
  from sklearn.cluster import MiniBatchKMeans
  from keras.datasets import mnist

  (x_train, y_train), (x_test, y_test) = mnist.load_data()
  X = x_train.reshape(len(x_train),-1)
  Y = y_train

  # normalize the data to 0 - 1

  X = X.astype(float) / 255.

  print(X.shape)
  print(X[0].shape)
  n_digits = len(np.unique(y_test))
  print(n_digits)

  # Initialize KMeans model

  kmeans = MiniBatchKMeans(n_clusters = n_digits)

  # Fit the model to the training data

  kmeans.fit(X)

  kmeans.labels_
  return kmeans


def assign_kmeans(km=None,X=None):
  pass
  # For each of the points in X, assign one of the means
  # refer to predict() function of the KMeans in sklearn
  # write your code ...
  from keras.datasets import mnist

  (x_train, y_train), (x_test, y_test) = mnist.load_data()
  kmeans = build_kmeans()
  X = x_train.reshape(len(x_train),-1)
  Y = y_train
  X = X.astype(float) / 255.

  actual_labels = Y

  def infer_cluster_labels(kmeans, actual_labels):
    inferred_labels = {}

    for i in range(kmeans.n_clusters):

        # find index of points in cluster
        labels = []
        index = np.where(kmeans.labels_ == i)

        # append actual labels for each point in cluster
        labels.append(Y[index])

        # determine most common label
        if len(labels[0]) == 1:
            counts = np.bincount(labels[0])
        else:
            counts = np.bincount(np.squeeze(labels))

        # assign the cluster to a value in the inferred_labels dictionary
        if np.argmax(counts) in inferred_labels:
            # append the new number to the existing array at this slot
            inferred_labels[np.argmax(counts)].append(i)
        else:
            # create a new array in this slot
            inferred_labels[np.argmax(counts)] = [i]

        #print(labels)
        #print('Cluster: {}, label: {}'.format(i, np.argmax(counts)))
        
    return inferred_labels

  def infer_data_labels(X_labels, cluster_labels):
    # empty array of len(X)
    predicted_labels = np.zeros(len(X_labels)).astype(np.uint8)

    for i, cluster in enumerate(X_labels):
        for key, value in cluster_labels.items():
            if cluster in value:
                predicted_labels[i] = key
                
    return predicted_labels

  # test the infer_cluster_labels() and infer_data_labels() functions

  cluster_labels = infer_cluster_labels(kmeans, Y)
  X_clusters = kmeans.predict(X)
  predicted_labels = infer_data_labels(X_clusters, cluster_labels)
  print (predicted_labels[:20])
  print (Y[:20])
  return predicted_labels


def compare_clusterings(ypred_1=None,ypred_2=None):
  # refer to sklearn documentation for homogeneity, completeness and vscore
  from keras.datasets import mnist
  from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score
  (x_train, y_train), (x_test, y_test) = mnist.load_data()
  kmeans = build_kmeans()
  X = x_train.reshape(len(x_train),-1)
  Y = y_train
  X = X.astype(float) / 255.

  labels_true = Y
  labels_pred = assign_kmeans()
  h,c,v = 0,0,0 # you need to write your code to find proper values
  h = homogeneity_score(labels_true, labels_pred)
  c = completeness_score(labels_true, labels_pred)
  v = v_measure_score(labels_true, labels_pred)
  return h,c,v

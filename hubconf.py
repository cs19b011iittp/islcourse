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

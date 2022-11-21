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

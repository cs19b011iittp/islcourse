# -*- coding: utf-8 -*-
"""cs19b011 isl endsem.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1KitJzuBXPcgp7EH_qc7KksyFUogq1k-6
"""

import torch
from torch import nn
import torch.optim as optim
from sklearn.datasets import fetch_openml
from sklearn.datasets import load_digits
import sklearn.cluster as skl_cluster
from sklearn.datasets import load_digits
from sklearn.datasets import make_blobs

from sklearn.datasets import make_circles
from sklearn.metrics import homogeneity_score
from sklearn.metrics import completeness_score
from sklearn.metrics.cluster import v_measure_score

from sklearn.datasets import make_blobs, make_circles
from sklearn.cluster import KMeans
from sklearn.metrics import homogeneity_completeness_v_measure

def get_data_blobs(n_points=100):
  X, y = make_blobs(n_samples=n_points, centers=3, n_features=2,random_state=0)
  return X,y

def get_data_circles(n_points=100):
  X, y = make_circles(n_samples=n_points, random_state=0, factor=0.8)
  return X,y

def get_data_mnist():
  X, y = load_digits(return_X_y=True, as_frame=True)
  return X,y

def build_kmeans(X=None,k=10):
  km = skl_cluster.KMeans(n_clusters=k, random_state=0).fit(X)
  return km

def assign_kmeans(km=None,X=None):
  ypred = km.predict(X)
  return ypred

def compare_clusterings(ypred_1=None,ypred_2=None):
  h = homogeneity_score(ypred_1, ypred_2)
  c = completeness_score(ypred_1, ypred_2)
  v = v_measure_score(ypred_1, ypred_2)
  return h,c,v

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV

def build_lr_model(X=None, y=None):
  from sklearn.linear_model import LogisticRegression
  lr_model = LogisticRegression(random_state=0).fit(X, y)
  # write your code...
  # Build logistic regression, refer to sklearn
  return lr_model

def build_rf_model(X=None, y=None):
  from sklearn.ensemble import RandomForestClassifier
  rf_model = RandomForestClassifier(max_depth=4, random_state=0).fit(X, y)
  # write your code...
  # Build Random Forest classifier, refer to sklearn
  return rf_model

def get_metrics(model1=None,X=None,y=None):
  from sklearn.metrics import precision_recall_fscore_support
  from sklearn import metrics
  # Obtain accuracy, precision, recall, f1score, auc score - refer to sklearn metrics
  y_pred = model1.predict(X)
  acc, prec, rec, f1 = precision_recall_fscore_support(y,y_pred)
  y_pred_proba = model1.predict_proba(X)
  fpr, tpr, thresholds = metrics.roc_curve(y, y_pred, pos_label=2)
  auc = metrics.auc(fpr, tpr)
  # write your code here...
  return acc, prec, rec, f1, auc

def get_paramgrid_lr():  
  from sklearn.model_selection import GridSearchCV
  from sklearn.linear_model import LogisticRegression
  lr_param_grid = grid={"penalty":["l1","l2"]}# l1 lasso l2 ridge
  
  return lr_param_grid

def get_paramgrid_rf():
  from sklearn.model_selection import GridSearchCV
  from sklearn.ensemble import RandomForestClassifier  
  rf_param_grid = {'n_estimators': [1, 10, 100], 'max_depth': [1, 10, None], 'criterion':['gini', 'entropy']}
  return rf_param_grid

def perform_gridsearch_cv_multimetric(model1=None, param_grid=None, cv=5, X=None, y=None, metrics=['accuracy','roc_auc']):
  from sklearn.model_selection import GridSearchCV
  from sklearn.linear_model import LogisticRegression
  from sklearn.ensemble import RandomForestClassifier
  import math
  grid_search_cv = GridSearchCV(model1, param_grid, cv=cv)
  grid_search_cv.fit(X, y)
  
  return [grid_search_cv.best_score_]


def loss_fn(y_pred, y_actual):
  v = -(y_actual * torch.log(y_pred + 0.0001))
  v = torch.sum(v)
  return v

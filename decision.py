#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
Created on Wed Nov 13 00:08:15 2019
@author: joshualiu
@author: Gonggajiezan (George)
"""
import pandas as pd
import os
import statistics as stats
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from joshua_feature_module import feature_extraction

# Splitting
x_flourishing_train,x_flourishing_test,y_flourishing_train,y_flourishing_test = train_test_split(x_flourishing_dataset,y_flourishing_dataset,test_size = 0.2,random_state = 1)
x_panas_pos_train,x_panas_pos_test,y_panas_pos_train,y_panas_pos_test = train_test_split(x_panas_pos_dataset,y_panas_postive_dataset,test_size = 0.2,random_state = 1)
x_panas_neg_train,x_panas_neg_test,y_panas_neg_train,y_panas_neg_test = train_test_split(x_panas_neg_dataset,y_panas_negative_dataset,test_size = 0.2,random_state = 1)

# Decision tree model(cross validation)
clf_flourishing = GridSearchCV(tree.DecisionTreeClassifier(criterion="entropy", class_weight="balanced"),
                               {'max_depth':range(3,40), 'min_samples_split':range(2,40)}, cv=10)
clf_flourishing.fit(x_flourishing_train, y_flourishing_train)
clf_flourishing_best = clf_flourishing.best_estimator_
print(clf_flourishing.best_score_, clf_flourishing.best_params_)
y_flourishing_pred = clf_flourishing_best.predict(x_flourishing_test)
# Decision tree model
# clf_flourishing = tree.DecisionTreeClassifier(criterion="entropy", max_depth=3,min_samples_leaf=12) 

# clf_flourishing = clf_flourishing.fit(x_flourishing_train, y_flourishing_train)
# y_flourishing_pred = clf_flourishing.predict(x_flourishing_test)
y_scores = clf_flourishing_best.predict_proba(x_flourishing_test)
print(y_flourishing_test)
fpr, tpr, threshold = roc_curve(y_flourishing_test, y_scores[:, 1])
auc_score = auc(fpr, tpr)
print("AUC score: ", auc_score)
print("Accuracy: ", accuracy_score(y_flourishing_test, y_flourishing_pred))

clf_panas_pos = GridSearchCV(tree.DecisionTreeClassifier(criterion="entropy", class_weight="balanced"),{'max_depth':range(3,30), 'min_samples_split':range(2,40)},cv=10)
clf_panas_pos.fit(x_panas_pos_train, y_panas_pos_train)
clf_panas_pos_best = clf_panas_pos.best_estimator_
print(clf_panas_pos.best_score_, clf_panas_pos.best_params_)
y_panas_pos_pred = clf_panas_pos_best.predict(x_panas_pos_test)
# clf_panas_pos  = tree.DecisionTreeClassifier(criterion="entropy")

# clf_panas_pos = clf_panas_pos.fit(x_panas_pos_train, y_panas_pos_train)
# y_panas_pos_pred = clf_panas_pos.predict(x_panas_pos_test)
y_scores = clf_panas_pos_best.predict_proba(x_panas_pos_test)
fpr, tpr, threshold = roc_curve(y_panas_pos_test, y_scores[:, 1])
auc_score = auc(fpr, tpr)
print("AUC score: ", auc_score)
print("Accuracy: ", accuracy_score(y_panas_pos_test, y_panas_pos_pred))

clf_panas_neg = GridSearchCV(tree.DecisionTreeClassifier(criterion="entropy",class_weight="balanced"),{'max_depth':range(3,30), 'min_samples_split':range(2,40)},cv=10)
clf_panas_neg.fit(x_panas_neg_train, y_panas_neg_train)
clf_panas_neg_best = clf_panas_neg.best_estimator_
print(clf_panas_neg.best_score_, clf_panas_neg.best_params_)
y_panas_neg_pred = clf_panas_neg_best.predict(x_panas_neg_test)

y_scores = clf_panas_neg_best.predict_proba(x_panas_neg_test)
fpr, tpr, threshold = roc_curve(y_panas_neg_test, y_scores[:, 1])
auc_score = auc(fpr, tpr)
print("AUC score: ", auc_score)
print("Accuracy: ", accuracy_score(y_panas_neg_test, y_panas_neg_pred))
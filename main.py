#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 00:08:15 2019

@author: joshualiu
"""
import pandas as pd 
import os

import numpy as np
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
import joshua_feature_module as js



def validation_function(train_features,train_label,model):
    train_label = np.array(train_label)
    acc_sum = 0;
    for train_index, test_index in kf.split(train_features,train_label):
        X_train, X_test = train_features[train_index], train_features[test_index]
        y_train, y_test = train_label[train_index], train_label[test_index]
        model.fit(X_train, y_train) 
        predication = model.predict(X_test)
#        try:
#            fixed_score = roc_auc_score(y_test, predication)
#        except ValueError:
#            pass
        ###here I just used accuracy, as data samples small, sometimes precison or recall is zero.
        acc_sum = acc_sum+roc_auc_score(y_test, predication)
    return acc_sum/10
if __name__ == "__main__": 
    x_train_positive,x_train_negative,x_train_flurishing,y_train_pos_score,y_train_neg_score,y_train_flourishing =  js.feature_extraction()
    
    #######normalization 
    scaler = StandardScaler()
    x_train_positive=scaler.fit_transform(x_train_positive)
    x_train_negative=scaler.fit_transform(x_train_negative)
    x_train_flurishing=scaler.fit_transform(x_train_flurishing)
    positive_x_train,positive_x_test,positive_y_train,positive_y_test = train_test_split(x_train_positive,y_train_pos_score,test_size = 0.2,random_state = 1)
    negative_x_train,negative_x_test,negative_y_train,negative_y_test = train_test_split(x_train_negative,y_train_neg_score,test_size = 0.2,random_state = 1)
    flourishing_x_train,flourishing_x_test,flourishing_y_train,flourishing_y_test = train_test_split(x_train_flurishing,y_train_flourishing,test_size = 0.2,random_state = 1)
    
    ##positive
    validation_train_AUC_score = {}
    kf = StratifiedKFold(n_splits = 10)
    model_positive = SVC(kernel='linear',gamma='auto')
    acc_linear = validation_function(positive_x_train,positive_y_train,model_positive)
    validation_train_AUC_score['linear'] = (acc_linear);

    
    model_positive = SVC(kernel='rbf',gamma='auto')
    acc_rbf = validation_function(positive_x_train,positive_y_train,model_positive)
    validation_train_AUC_score['rbf'] = (acc_rbf);
    
    model_positive = SVC(kernel='poly',gamma='auto')
    acc_poly = validation_function(positive_x_train,positive_y_train,model_positive)
    validation_train_AUC_score['poly'] = (acc_poly);
    
    
    best_kernal = max(validation_train_AUC_score, key=validation_train_AUC_score.get)
    model_positive = SVC(kernel=best_kernal,gamma='auto')
    model_positive.fit(positive_x_train,positive_y_train)
    positive_y_test_pred = model_positive.predict(positive_x_test)
    print('auc roc of SVM on Panas positive:',roc_auc_score(positive_y_test, positive_y_test_pred))
        
    ##negative
    validation_train_AUC_score = {}
    kf = StratifiedKFold(n_splits = 10)
    model_negative = SVC(kernel='linear',gamma='auto')
    acc_linear = validation_function(negative_x_train,negative_y_train,model_negative)
    validation_train_AUC_score['linear'] = (acc_linear);

    
    model_negative = SVC(kernel='rbf',gamma='auto')
    acc_rbf = validation_function(negative_x_train,negative_y_train,model_negative)
    validation_train_AUC_score['rbf'] = (acc_rbf);
    
    model_negative = SVC(kernel='poly',gamma='auto')
    acc_poly = validation_function(negative_x_train,negative_y_train,model_negative)
    validation_train_AUC_score['poly'] = (acc_poly);
    
    
    best_kernal = max(validation_train_AUC_score, key=validation_train_AUC_score.get)
    model_positive = SVC(kernel=best_kernal,gamma='auto')
    model_positive.fit(negative_x_train,negative_y_train)
    negative_y_test_pred = model_positive.predict(negative_x_test)
    print('auc roc of SVM on Panas negative:',roc_auc_score(negative_y_test, negative_y_test_pred))
##    
##    #flourishing
    validation_train_AUC_score = {}
    kf = StratifiedKFold(n_splits = 10)
    model_flourishing = SVC(kernel='linear',gamma='auto')
    acc_linear = validation_function(flourishing_x_train,flourishing_y_train,model_flourishing)
    validation_train_AUC_score['linear'] = (acc_linear);

    
    model_flourishing = SVC(kernel='rbf',gamma='auto')
    acc_rbf = validation_function(flourishing_x_train,flourishing_y_train,model_flourishing)
    validation_train_AUC_score['rbf'] = (acc_rbf);
    
    model_flourishing = SVC(kernel='sigmoid',gamma='auto')
    acc_poly = validation_function(flourishing_x_train,flourishing_y_train,model_flourishing)
    validation_train_AUC_score['poly'] = (acc_poly);
    
    
    best_kernal = max(validation_train_AUC_score, key=validation_train_AUC_score.get)
    model_flourishing = SVC(kernel=best_kernal,gamma='auto')
    model_flourishing.fit(flourishing_x_train,flourishing_y_train)
    flourishing_y_test_pred = model_flourishing.predict(flourishing_x_test)
    print('auc roc of SVM on flourishing:',roc_auc_score(flourishing_y_test, flourishing_y_test_pred))
##    
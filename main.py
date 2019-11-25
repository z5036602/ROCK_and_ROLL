import pandas as pd
import os
import statistics as stats
import numpy as np
from numpy import vstack, array, nan
from numpy import vstack, array, nan
from sklearn.impute import SimpleImputer as si
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from sklearn.decomposition import PCA
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score

imp_module = 'module-drop'
imp_function = 'feature_extraction'

ip_module = __import__(imp_module)
data_extraction = getattr(ip_module, imp_function)
x_train_positive, x_train_negative, x_train_flurishing, y_train_pos_score, y_train_neg_score, y_train_flourishing = data_extraction()
# print(x_train_positive.shape)
# print(x_train_negative.shape)
# print(x_train_flurishing.shape)

scaler = StandardScaler()
x_train_positive = scaler.fit_transform(x_train_positive)
x_train_negative = scaler.fit_transform(x_train_negative)
x_train_flurishing = scaler.fit_transform(x_train_flurishing)


# pca = PCA(n_components=6)
# x_train_positive = pca.fit_transform(x_train_positive)
# x_train_negative = pca.fit_transform(x_train_negative)
# x_train_flurishing = pca.fit_transform(x_train_flurishing)
positive_x_train, positive_x_test, positive_y_train, positive_y_test = train_test_split(x_train_positive,
                                                                                        y_train_pos_score,
                                                                                        test_size=0.2,
                                                                                        random_state=1)
negative_x_train, negative_x_test, negative_y_train, negative_y_test = train_test_split(x_train_negative,
                                                                                        y_train_neg_score,
                                                                                        test_size=0.2,
                                                                                        random_state=1)
flourishing_x_train, flourishing_x_test, flourishing_y_train, flourishing_y_test = train_test_split(
    x_train_flurishing, y_train_flourishing, test_size=0.2, random_state=1)


def cross_val_and_build_model(x_train, y_train):
    k_scores = []
    for k in range(1,20):
        knn = KNeighborsClassifier(n_neighbors = k, weights = 'distance')
        cv = StratifiedKFold(n_splits=5)
        score = cross_val_score(knn, x_train, np.ravel(y_train), cv=cv, scoring='roc_auc').mean()
        k_scores.append(score)
    print(np.argmax(k_scores)+1, max(k_scores))
    return np.argmax(k_scores) + 1


# get best k
best_flourishing_k = cross_val_and_build_model(flourishing_x_train, flourishing_y_train)
best_panas_pos_k = cross_val_and_build_model(positive_x_train, positive_y_train)
best_panas_neg_k = cross_val_and_build_model(negative_x_train, negative_y_train)

# get test accuracy
model_flour = KNeighborsClassifier(n_neighbors=best_flourishing_k, weights='distance')
model_pos = KNeighborsClassifier(n_neighbors=best_panas_pos_k, weights='distance')
model_neg = KNeighborsClassifier(n_neighbors=best_panas_neg_k, weights='distance')

model_flour.fit(flourishing_x_train, np.ravel(flourishing_y_train))
model_pos.fit(positive_x_train, np.ravel(positive_y_train))
model_neg.fit(negative_x_train, np.ravel(negative_y_train))

'''for k in range(1,31):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(train_x, train_y)
    train_y_predicted = model.predict_proba(train_x)[:,1]
    test_y_predicted = model.predict_proba(test_x)[:,1]
    # false_positive_rate, true_positive_rate, thresholds = roc_curve(test_y, test_y_predict)
    train_auc_result.append(roc_auc_score(train_y, train_y_predicted))
    test_auc_result.append(roc_auc_score(test_y, test_y_predicted))'''



prediction_flour = model_flour.predict_proba(flourishing_x_test)[:,1]
prediction_pos = model_pos.predict_proba(positive_x_test)[:,1]
prediction_neg = model_neg.predict_proba(negative_x_test)[:,1]

accuracy_flour = roc_auc_score(np.ravel(flourishing_y_test), prediction_flour)
accuracy_pos = roc_auc_score(np.ravel(positive_y_test), prediction_pos)
accuracy_neg = roc_auc_score(np.ravel(negative_y_test), prediction_neg)
print(accuracy_flour, accuracy_pos, accuracy_neg)


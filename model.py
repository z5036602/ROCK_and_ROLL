import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from data_extractor import *

def normalize_feature(df, feature_name):
    df[feature_name] = (df[feature_name] - min(df[feature_name])) / (max(df[feature_name]) - min(df[feature_name]))
    return None

def convert_to_binary_label(df, feature_name):
    median = df[feature_name].median()

    zero_index_list = df[feature_name] < median
    one_index_list = df[feature_name] >= median 

    df.loc[zero_index_list, feature_name] = 0
    df.loc[one_index_list, feature_name] = 1
    return None

def get_features_and_labels(df):
    feature_data = df.iloc[:, 1:]
    label_data = df.iloc[:, 0]
    return feature_data, label_data    

def add_input_features(df):
    input_data_path = os.path.join(dataset_path, 'StudentLife_Dataset/Inputs/sensing')
    df = add_feature_to_dataframe(df, *get_activity_time(os.path.join(input_data_path, 'activity/')))    
    df = add_feature_to_dataframe(df, *get_audio_time(os.path.join(input_data_path, 'audio/')))  
    df = add_feature_to_dataframe(df, *get_conversation_data(os.path.join(input_data_path, 'conversation/')))   
    df = add_feature_to_dataframe(df, *get_dark_time_data(os.path.join(input_data_path, 'dark/')))
    df = add_feature_to_dataframe(df, *get_phone_charge_data(os.path.join(input_data_path, 'phonecharge/')))
    df = add_feature_to_dataframe(df, *get_phone_lock_data(os.path.join(input_data_path, 'phonelock/')))
    df = add_feature_to_dataframe(df, *get_bluetooth_data(os.path.join(input_data_path, 'bluetooth/')))
    df = add_feature_to_dataframe(df, *get_indoor_outdoor_data(os.path.join(input_data_path, 'wifi_location/')))

    print(df.describe())

    return df

def add_feature_to_dataframe(df, features, feature_names):
    df_new = pd.DataFrame.from_dict(features, orient='index')
    df_new.columns = feature_names
    df = df.join(df_new)
    return df

def predict_flourishing_score():
    print('Predict Flourishing Score')

    processed_data_file_name = 'flourishing_score_data.csv'


    if not os.path.exists(processed_data_file_name):
        flourishing_score_diff = get_flourishing_score(dataset_path)
        #taking only flourishing_score_diff for now
        df = pd.DataFrame.from_dict(flourishing_score_diff, orient='index')
        df.columns = ['fl_score']
        df = add_input_features(df)
        df.to_csv(processed_data_file_name, header=True)
    else:
        print('did not read original files')
        df = pd.read_csv(processed_data_file_name, index_col = 0)

    print(df)

    convert_to_binary_label(df, 'fl_score')

    print(df)

    for feature in df.columns[1:]:
        normalize_feature(df, feature)


    feature_data, label_data = get_features_and_labels(df)


    train_feature_data, test_feature_data, train_label_data, test_label_data = train_test_split(feature_data, label_data, test_size = 0.2, random_state = 1)
    

    model = LogisticRegression(solver = 'liblinear')
    model.fit(train_feature_data, train_label_data)

    train_pred = model.predict(train_feature_data)
    train_pred_prob = np.array([val[1] for val in model.predict_proba(train_feature_data)])  
    test_pred = model.predict(test_feature_data)
    test_pred_prob = np.array([val[1] for val in model.predict_proba(test_feature_data)])

    get_metric_scores(train_label_data, train_pred_prob, train_pred, test_label_data, test_pred_prob, test_pred)

    print('****************************************')


def predict_panas_pos():
    print('Predict Panas Positive')

    processed_data_file_name = 'panas_pos_score_data.csv'

    if not os.path.exists(processed_data_file_name):
        panas_positive_diff, _ = get_panas_score(dataset_path)
        #taking only flourishing_score_diff for now
        df = pd.DataFrame.from_dict(panas_positive_diff, orient='index')
        df.columns = ['pos_panas']
        df = add_input_features(df)
        df.to_csv(processed_data_file_name, header=True)
    else:
        print('did not read original files')
        df = pd.read_csv(processed_data_file_name, index_col = 0)

    convert_to_binary_label(df, 'pos_panas')

    for feature in df.columns[1:]:
        normalize_feature(df, feature)

    feature_data, label_data = get_features_and_labels(df)

    train_feature_data, test_feature_data, train_label_data, test_label_data = train_test_split(feature_data, label_data, test_size = 0.2, random_state = 1)

    model = LogisticRegression(solver = 'liblinear')
    model.fit(train_feature_data, train_label_data)

    train_pred = model.predict(train_feature_data)
    train_pred_prob = np.array([val[1] for val in model.predict_proba(train_feature_data)])  
    test_pred = model.predict(test_feature_data)
    test_pred_prob = np.array([val[1] for val in model.predict_proba(test_feature_data)])

    get_metric_scores(train_label_data, train_pred_prob, train_pred, test_label_data, test_pred_prob, test_pred)

    print('****************************************')


def predict_panas_neg():
    print('Predict Panas Negative')

    processed_data_file_name = 'panas_neg_score_data.csv'

    if not os.path.exists(processed_data_file_name):
        _, panas_negative_diff = get_panas_score(dataset_path)
        #taking only flourishing_score_diff for now
        df = pd.DataFrame.from_dict(panas_negative_diff, orient='index')
        df.columns = ['neg_panas']
        df = add_input_features(df)
        df.to_csv(processed_data_file_name, header=True)
    else:
        print('did not read original files')
        df = pd.read_csv(processed_data_file_name, index_col = 0)

    convert_to_binary_label(df, 'neg_panas')

    for feature in df.columns[1:]:
        normalize_feature(df, feature)

    feature_data, label_data = get_features_and_labels(df)

    train_feature_data, test_feature_data, train_label_data, test_label_data = train_test_split(feature_data, label_data, test_size = 0.2, random_state = 1)


    model = LogisticRegression(solver = 'liblinear')
    model.fit(train_feature_data, train_label_data)

    train_pred = model.predict(train_feature_data)
    train_pred_prob = np.array([val[1] for val in model.predict_proba(train_feature_data)])  
    test_pred = model.predict(test_feature_data)
    test_pred_prob = np.array([val[1] for val in model.predict_proba(test_feature_data)])

    get_metric_scores(train_label_data, train_pred_prob, train_pred, test_label_data, test_pred_prob, test_pred)

    print('****************************************')


def get_metric_scores(train_label_data, train_pred_prob, train_pred, test_label_data, test_pred_prob, test_pred):
    auc = roc_auc_score(train_label_data, train_pred_prob)
    print('Train AUC : ', auc)
    auc = roc_auc_score(test_label_data, test_pred_prob)
    print('Test AUC : ', auc)


def plot_feature_vs_label(feature_data, label):
    pass
    
if __name__ == '__main__':
    predict_flourishing_score()

    predict_panas_pos()

    predict_panas_neg()



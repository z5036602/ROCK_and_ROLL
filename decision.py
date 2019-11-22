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


##get acitivity data, remove unknow data
def get_activity_data(file_path):
    all_path = os.listdir(file_path)
    activity_data = {}
    fea = []
    acitivity_ratio_list = []
    user_name_list = []
    for path in all_path:
        csv_full_path = os.path.join(file_path, path)
        df = pd.read_csv(csv_full_path, index_col=False)
        all_feats = df[' activity inference'].value_counts()
        all_counts = df[' activity inference'].count()
        zero_count_ratio = all_feats[0] / all_counts
        one_count_ratio = all_feats[1] / all_counts
        two_count_ratio = all_feats[2] / all_counts
        three_count_ratio = all_feats[3] / all_counts
        # print(df)
        user_name = path.split('_')[1].replace('.csv', '')
        fea = [zero_count_ratio, one_count_ratio, two_count_ratio,three_count_ratio]
        activity_data[user_name] = fea
    return activity_data


##get audio data, remove unknown
def get_audio_data(file_path):
    all_path = os.listdir(file_path)
    audio_feats = {}
    fea = []
    # acitivity_ratio_list = []
    # user_name_list = []
    for path in all_path:
        csv_full_path = os.path.join(file_path, path)
        df = pd.read_csv(csv_full_path, index_col=False)
        all_feats = df[' audio inference'].value_counts()
        all_counts = df[' audio inference'].count()
        zero_count_ratio = all_feats[0] / all_counts
        one_count_ratio = all_feats[1] /  all_counts
        two_count_ratio = all_feats[2] / all_counts
        fea = [zero_count_ratio, one_count_ratio, two_count_ratio]

        # print(all_feats[2])
        user_name = path.split('_')[1].replace('.csv', '')
        audio_feats[user_name] = fea;
    return audio_feats


def get_conversation_data(file_path):
    all_path = os.listdir(file_path)
    conv_feats = {}
    fea = []
    for path in all_path:
        csv_full_path = os.path.join(file_path, path)
        df = pd.read_csv(csv_full_path, index_col=False)
        duration = df[' end_timestamp'] - df['start_timestamp']
        total_dur = sum(duration)
        freq_conv = df['start_timestamp'].count()
        average = total_dur / freq_conv
        variance = duration.var()
        fea = [total_dur, freq_conv, average, variance]
        user_name = path.split('_')[1].replace('.csv', '')
        conv_feats[user_name] = fea;
    return conv_feats;


def get_dark(file_path):
    all_path = os.listdir(file_path)
    conv_feats = {}
    fea = []
    # acitivity_ratio_list = []
    # user_name_list = []
    for path in all_path:
        csv_full_path = os.path.join(file_path, path)
        df = pd.read_csv(csv_full_path, index_col=False)
        duration = df['end'] - df['start']
        total_dur = sum(duration)
        freq_dark = df['end'].count()
        average = total_dur / freq_dark
        variance = duration.var()

        fea = [total_dur, freq_dark, average, variance]

        user_name = path.split('_')[1].replace('.csv', '')
        conv_feats[user_name] = fea;
    return conv_feats;


def get_phone_lock(file_path):
    all_path = os.listdir(file_path)
    conv_feats = {}
    fea = []
    for path in all_path:
        csv_full_path = os.path.join(file_path, path)
        df = pd.read_csv(csv_full_path, index_col=False)
        duration = df['end'] - df['start']
        total_dur = sum(duration)
        freq_conv = df['end'].count()
        average = total_dur/ freq_conv
        variance = duration.var()
        fea = [total_dur, freq_conv, average, variance]
        user_name = path.split('_')[1].replace('.csv', '')
        conv_feats[user_name] = fea;
    return conv_feats;


#GPS not useful, missing value for all gps, travelsate; bearing, speed too many zeros;
def get_GPS(file_path):
    all_path = os.listdir(file_path)
    conv_feats = {}
    fea = []
    for path in all_path:
        csv_full_path = os.path.join(file_path, path)
        df = pd.read_csv(csv_full_path,index_col=False, converters = {'travelstate': str})
        new_df = df.dropna(axis=0, how='any', inplace=False)
        freq = len(new_df['time'])
        network = len(new_df['provider']) / len(df)
        fea = [freq, network]
        user_name = path.split('_')[1].replace('.csv', '')
        conv_feats[user_name] = fea;
    return conv_feats




def get_bluetooth(file_path):
    all_path = os.listdir(file_path)
    conv_feats = {}
    fea = []
    # acitivity_ratio_list = []
    # user_name_list = []
    for path in all_path:
        csv_full_path = os.path.join(file_path, path)
        df = pd.read_csv(csv_full_path, index_col=False)
        MAC_fre = df['MAC'].value_counts()
        top_ten_mac = sum(MAC_fre[:10])
        top_mac_ratio = top_ten_mac / sum(MAC_fre)
        class_id_fre = df['class_id'].value_counts()
        top_ten_class = sum(class_id_fre[:5])
        top_class_ratio = top_ten_class / sum(class_id_fre)

        MAC_encountered = len(df['MAC'].unique())
        class_encountered = len(df['class_id'].unique())

        fea = [top_mac_ratio, top_class_ratio, MAC_encountered, class_encountered]
        user_name = path.split('_')[1].replace('.csv', '')
        # print("user name:", user_name)
        conv_feats[user_name] = fea;

    return conv_feats


def get_wifi(file_path):
    all_path = os.listdir(file_path)
    conv_feats = {}
    fea = []
    for path in all_path:
        csv_full_path = os.path.join(file_path, path)
        df = pd.read_csv(csv_full_path, index_col=False)
        BSSID_fre = df['BSSID'].value_counts()
        top_ten_bssid = sum(BSSID_fre[:5])
        top_bssid_ratio = top_ten_bssid / sum(BSSID_fre)
        pre_id_fre = df['freq'].value_counts()
        top_ten_class = sum(pre_id_fre[:5])
        top_class_ratio = top_ten_class / sum(pre_id_fre)

        BSSID_encountered = len(df['BSSID'].unique())
        fre_encountered = len(df['freq'].unique())

        fea = [top_bssid_ratio, top_class_ratio, BSSID_encountered, fre_encountered]
        user_name = path.split('_')[1].replace('.csv', '')
        conv_feats[user_name] = fea

    return conv_feats



def get_wifi_location(file_path):
    all_path = os.listdir(file_path)
    conv_feats = {}
    fea = []
    for path in all_path:
        csv_full_path = os.path.join(file_path, path)
        df = pd.read_csv(csv_full_path, index_col=False)
        total_length = len(df['time'])
        count_in = 0
        count_near = 0
        for item in df['location']:
            if ('in[' in item):
                count_in += 1
            if ('near[' in item):
                count_near += 1
                # if df['time']

        count_in_ratio = count_in / total_length
        count_near_ratio = count_near / total_length
        location_encountered = len(df['location'].unique())

        fea = [count_in_ratio, count_near_ratio, total_length, location_encountered]
        user_name = path.split('_')[2].replace('.csv', '')
        conv_feats[user_name] = fea

    return conv_feats


##flourishing score: $post-pre%
def Flourishing_score(file_path):
    # pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    csv_full_path = os.path.join(file_path, 'FlourishingScale.csv')
    post_flourishing_score_dictionary = {}
    pre_flourishing_score_dictionary = {}
    flourishing_score_dictionary = []

    my_list = []
    df = pd.read_csv(csv_full_path, index_col=False)
    new_numpy_matrix = df.dropna(axis=0, how='any', inplace=False).to_numpy()
    # user_counts = numpy_matrix['uid'].unique()
    # remove_list = ['u51', 'u18', 'u57', 'u13', 'u00', 'u47', 'u22', 'u39', 'u12', 'u50', 'u58']
    # rest = list(set(user_counts) ^ set(remove_list))
    # new_numpy_matrix = numpy_matrix[numpy_matrix['uid'].isin(rest)].to_numpy()
    list_flourishing_score = np.sum(new_numpy_matrix[:, 2:10], 1)
    post_index = np.where(new_numpy_matrix[:, 1] == 'post')
    pre_index = np.where(new_numpy_matrix[:, 1] == 'pre')
    flourishing_score_change = {}
    for index in np.nditer(post_index):
        key = new_numpy_matrix[index, 0]
        post_flourishing_score_dictionary[key] = list_flourishing_score[index]
    for index in np.nditer(pre_index):
        key = new_numpy_matrix[index, 0]
        pre_flourishing_score_dictionary[key] = list_flourishing_score[index]

    if len(post_flourishing_score_dictionary)>= len(pre_flourishing_score_dictionary):
        for key in post_flourishing_score_dictionary:
            if key not in pre_flourishing_score_dictionary:
                continue;
            change = post_flourishing_score_dictionary[key] - pre_flourishing_score_dictionary[key]
            if change >= 0:
                label = 'positive'
            else:
                label = 'negative'
            my_list = [label]
            flourishing_score_change[key] = my_list
    else:
        for key in pre_flourishing_score_dictionary:
            if key not in post_flourishing_score_dictionary:
                continue;
            change = post_flourishing_score_dictionary[key] - pre_flourishing_score_dictionary[key]
            if change >= 0:
                label = 'positive'
            else:
                label = 'negative'
            my_list = [label]
            flourishing_score_change[key] = my_list
    # print(flourishing_score_change)
    return flourishing_score_change



##panas score: $post-pre% for postive and negative
def panas_score(file_path):
    csv_full_path = os.path.join(file_path, 'panas.csv')
    post_pos_panas_score_dictionary = {}
    pre_pos_panas_score_dictionary = {}
    post_neg_panas_score_dictionary = {}
    pre_neg_panas_score_dictionary = {}
    panas_positive_score_dictionary = {}
    panas_negative_score_dictionary = {}
    my_list = []

    df = pd.read_csv(csv_full_path, index_col=False)
    numpy_matrix = df.dropna(axis=0, how='any', inplace=False).to_numpy()
    list_positive_score = np.sum(numpy_matrix[:, [2, 5, 9, 10, 12, 13, 15, 16, 18]], 1)
    list_negative_score = np.sum(numpy_matrix[:, [3, 4, 6, 7, 8, 11, 14, 17, 19]], 1)
    post_index = np.where(numpy_matrix[:, 1] == 'post')
    pre_index = np.where(numpy_matrix[:, 1] == 'pre')
    # print(pre_index,post_index)
    for index in np.nditer(post_index):
        key = numpy_matrix[index, 0]
        post_pos_panas_score_dictionary[key] = list_positive_score[index]
        post_neg_panas_score_dictionary[key] = list_negative_score[index]
    for index in np.nditer(pre_index):
        key = numpy_matrix[index, 0]
        pre_pos_panas_score_dictionary[key] = list_positive_score[index]
        pre_neg_panas_score_dictionary[key] = list_negative_score[index]

    if len(post_pos_panas_score_dictionary) >= len(pre_pos_panas_score_dictionary):
        for key in post_pos_panas_score_dictionary:
            if key not in pre_pos_panas_score_dictionary:
                continue;
            pos_change = post_pos_panas_score_dictionary[key] - pre_pos_panas_score_dictionary[key]
            neg_change = post_neg_panas_score_dictionary[key] - pre_neg_panas_score_dictionary[key]
            if pos_change >= 0:
                label = 'positive'
            else:
                label = 'negative'
            my_list = [label]
            panas_positive_score_dictionary[key] = my_list
            if neg_change >= 0:
                label = 'positive'
            else:
                label = 'negative'
            my_list = [label]
            panas_negative_score_dictionary[key] = my_list
    else:
        for key in pre_pos_panas_score_dictionary:
            if key not in post_pos_panas_score_dictionary:
                continue;
            pos_change = post_pos_panas_score_dictionary[key] - pre_pos_panas_score_dictionary[key]
            neg_change = post_neg_panas_score_dictionary[key] - pre_neg_panas_score_dictionary[key]
            if pos_change >= 0:
                label = 'positive'
            else:
                label = 'negative'
            my_list = [label]
            panas_positive_score_dictionary[key] = my_list
            if neg_change >= 0:
                label = 'negative'
            else:
                label = 'positive'
            my_list = [label]
            panas_negative_score_dictionary[key] = my_list
            # my_list = [pos_change]
            # panas_positive_score_dictionary[key] = my_list
            # my_list = [neg_change]
            # panas_negative_score_dictionary[key] = my_list

    ##
    ##        x_train.append(fea)

    return panas_positive_score_dictionary, panas_negative_score_dictionary


# In[2]:


d_activity = get_activity_data("StudentLife_Dataset/Inputs/sensing/activity")
d_audio = get_audio_data("StudentLife_Dataset/Inputs/sensing/audio")
d_conversation = get_conversation_data("StudentLife_Dataset/Inputs/sensing/conversation")
d_dark = get_dark("StudentLife_Dataset/Inputs/sensing/dark")
d_phone_lock = get_phone_lock("StudentLife_Dataset/Inputs/sensing/phonelock")
d_phone_charge = get_phone_lock("StudentLife_Dataset/Inputs/sensing/phonecharge")
d_gps = get_GPS("StudentLife_Dataset/Inputs/sensing/gps")
d_wifi = get_wifi("StudentLife_Dataset/Inputs/sensing/wifi")
d_wifi_location = get_wifi_location("StudentLife_Dataset/Inputs/sensing/wifi_location")
d_get_bluetooth = get_bluetooth("StudentLife_Dataset/Inputs/sensing/bluetooth")
flourishing_score = Flourishing_score("StudentLife_Dataset/Outputs")
positive_score, negative_score = panas_score("StudentLife_Dataset/Outputs")
print("finished")


# In[17]:


#flourishing features and labels
x_train_flurishing = []
y_flourishing = []
for i in range(60):
    u = "u{:02d}".format(i)
    if u not in flourishing_score:
        continue;
    fea = d_activity[u] + d_audio[u] + d_conversation[u] + d_dark[u] + d_phone_lock[u]           +d_phone_charge[u] + d_gps[u] + d_wifi[u] + d_wifi_location[u] + d_get_bluetooth[u]
    y_flourishing.append(flourishing_score[u])
    x_train_flurishing.append(fea)
x_flourishing_dataset = np.array(x_train_flurishing)
y_flourishing_dataset = np.array(y_flourishing)


# panas features and labels
x_dataset = []
y_positive = []
y_negative = []
for i in range(60):
    u = "u{:02d}".format(i)
    if u not in positive_score:
        continue;
    fea = d_activity[u] + d_audio[u] + d_conversation[u] + d_dark[u] + d_phone_lock[u]           + d_phone_charge[u] + d_gps[u] + d_wifi[u] + d_wifi_location[u] + d_get_bluetooth[u]
    x_dataset.append(fea)
    y_positive.append(positive_score[u])
    y_negative.append(negative_score[u])

x_panas_dataset = np.array(x_dataset)
y_panas_postive_dataset = np.array(y_positive)
y_panas_negative_dataset = np.array(y_negative)

print("hello")
print(len(x_flourishing_dataset))
print(len(y_flourishing_dataset))

#normalizaiton
# scaler = StandardScaler()
# x_flourishing_dataset=scaler.fit_transform(x_flourishing_dataset)
# x_panas_dataset=scaler.fit_transform(x_panas_dataset)

# Decomposition
pca = PCA(n_components=10)
x_flourishing_dataset = pca.fit_transform(x_flourishing_dataset)
pca1 = PCA(n_components=8)
x_panas_dataset = pca1.fit_transform(x_panas_dataset)
# Splitting
x_flourishing_train,x_flourishing_test,y_flourishing_train,y_flourishing_test = train_test_split(x_flourishing_dataset,y_flourishing_dataset,test_size = 0.2,random_state = 1)
x_panas_pos_train,x_panas_pos_test,y_panas_pos_train,y_panas_pos_test = train_test_split(x_panas_dataset,y_panas_postive_dataset,test_size = 0.2,random_state = 1)
x_panas_neg_train,x_panas_neg_test,y_panas_neg_train,y_panas_neg_test = train_test_split(x_panas_dataset,y_panas_negative_dataset,test_size = 0.2,random_state = 1)


# Decision tree model(cross validation)
clf_flourishing = GridSearchCV(tree.DecisionTreeClassifier(criterion="entropy"),{'max_depth':range(3,30), 'min_samples_leaf':range(5,40)},cv=10)
clf_flourishing.fit(x_flourishing_train, y_flourishing_train)
clf_flourishing_best = clf_flourishing.best_estimator_
print(clf_flourishing.best_score_, clf_flourishing.best_params_)
y_flourishing_pred = clf_flourishing_best.predict(x_flourishing_test)
# Decision tree model
# clf_flourishing = tree.DecisionTreeClassifier(criterion="entropy", max_depth=3,min_samples_leaf=12) 

# clf_flourishing = clf_flourishing.fit(x_flourishing_train, y_flourishing_train)
# y_flourishing_pred = clf_flourishing.predict(x_flourishing_test)
print("Accuracy: ", accuracy_score(y_flourishing_test, y_flourishing_pred))


# In[18]:



clf_panas_pos = GridSearchCV(tree.DecisionTreeClassifier(criterion="entropy"),{'max_depth':range(3,30), 'min_samples_leaf':range(5,40)},cv=10)
clf_panas_pos.fit(x_panas_pos_train, y_panas_pos_train)
clf_panas_pos_best = clf_panas_pos.best_estimator_
print(clf_panas_pos.best_score_, clf_panas_pos.best_params_)
y_panas_pos_pred = clf_panas_pos_best.predict(x_panas_pos_test)
# clf_panas_pos  = tree.DecisionTreeClassifier(criterion="entropy")

# clf_panas_pos = clf_panas_pos.fit(x_panas_pos_train, y_panas_pos_train)
# y_panas_pos_pred = clf_panas_pos.predict(x_panas_pos_test)
print("Accuracy: ", accuracy_score(y_panas_pos_test, y_panas_pos_pred))


# In[20]:


clf_panas_neg = GridSearchCV(tree.DecisionTreeClassifier(criterion="entropy",class_weight="balanced"),{'max_depth':range(3,30), 'min_samples_leaf':range(5,40) },cv=10)
clf_panas_neg.fit(x_panas_neg_train, y_panas_neg_train)
clf_panas_neg_best = clf_panas_neg.best_estimator_
print(clf_panas_neg.best_score_, clf_panas_neg.best_params_)
y_panas_neg_pred = clf_panas_neg_best.predict(x_panas_neg_test)
# clf_panas_neg  = tree.DecisionTreeClassifier(criterion="entropy")
# clf_panas_neg  = clf_panas_neg.fit(x_panas_neg_train, y_panas_pos_train)
# y_panas_neg_pred  = clf_panas_neg.predict(x_panas_neg_test)
print("Accuracy: ", accuracy_score(y_panas_neg_test, y_panas_neg_pred))


# In[ ]:





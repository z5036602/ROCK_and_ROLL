#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 00:08:15 2019
@author: joshualiu
"""
import pandas as pd
import os
import statistics as stats
import numpy as np


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
        exit()
        all_counts = df[' audio inference'].count()
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
        one_count_ratio = all_feats[1] / all_counts
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
        fea = [total_dur, freq_conv, conv_time, variance]
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
        conv_feats[user_name] = fea;

    return conv_feats


##flourishing score: $post-pre%
def Flourishing_score(file_path):
    csv_full_path = os.path.join(file_path, 'FlourishingScale.csv')
    post_flourishing_score_dictionary = {}
    pre_flourishing_score_dictionary = {}
    flourishing_score_dictionary = []

    my_list = []
    df = pd.read_csv(csv_full_path)
    print(df)
    exit()
    new_df = df.dropna(axis=0, how='any', inplace=False)
    numpy_matrix = df.fillna(df.mean()).to_numpy()
    list_flourishing_score = np.sum(numpy_matrix[:, 2:9], 1)
    post_index = np.where(numpy_matrix[:, 1] == 'post')
    pre_index = np.where(numpy_matrix[:, 1] == 'pre')
    flourishing_score_change = {}
    for index in np.nditer(post_index):
        key = numpy_matrix[index, 0]
        post_flourishing_score_dictionary[key] = list_flourishing_score[index]
        # print('key_is:',key)

    for index in np.nditer(pre_index):
        key = numpy_matrix[index, 0]
        pre_flourishing_score_dictionary[key] = list_flourishing_score[index]
        # print('key_is:',key)
    for key in post_flourishing_score_dictionary:

        #        u="u{:02d}".format(i)
        if key not in pre_flourishing_score_dictionary:
            continue;
        change = post_flourishing_score_dictionary[key] - pre_flourishing_score_dictionary[key]
        my_list = [change]
        flourishing_score_change[key] = my_list
    #        fea=d_activity[u]+d_audio[u]+d_conversation[u]+d_dark[u]+d_phone_lock[u] \
    #        +d_innout[u]+d_get_bluetooth[u]
    # print(type(flourishing_score_change[key]))
    #        x_train.append(fea)
    # print(type(flourishing_score_change[key]))
    print(flourishing_score_change)
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

    df = pd.read_csv(csv_full_path)
    numpy_matrix = df.fillna(df.mean()).to_numpy()
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

    for key in post_pos_panas_score_dictionary:
        if key not in pre_pos_panas_score_dictionary:
            continue;
        pos_change = post_pos_panas_score_dictionary[key] - pre_pos_panas_score_dictionary[key]
        neg_change = post_neg_panas_score_dictionary[key] - pre_neg_panas_score_dictionary[key]
        my_list = [pos_change]
        panas_positive_score_dictionary[key] = my_list
        my_list = [neg_change]
        panas_negative_score_dictionary[key] = my_list

    ##
    ##        x_train.append(fea)

    return panas_positive_score_dictionary, panas_negative_score_dictionary


if __name__ == "__main__":

    # print(Flourishing_score("/Users/joshualiu/Desktop/GroupProject/StudentLife_Dataset/Outputs"))
    # print(panas_score("/Users/joshualiu/Desktop/GroupProject/StudentLife_Dataset/Outputs"))
    # d_activity = get_activity_data("/Users/chris/Documents/9417/project/StudentLife_Dataset/Inputs/sensing/activity")
    # d_audio = get_audio_data("/Users/chris/Documents/9417/project/StudentLife_Dataset/Inputs/sensing/audio")
    # d_conversation = get_conversation_data("/Users/chris/Documents/9417/project/StudentLife_Dataset/Inputs/sensing/conversation")
    # d_dark = get_dark("/Users/chris/Documents/9417/project/StudentLife_Dataset/Inputs/sensing/dark")
    # d_phone_lock = get_phone_lock("/Users/chris/Documents/9417/project/StudentLife_Dataset/Inputs/sensing/phonelock")
    # d_phone_charge = get_phone_lock("/Users/chris/Documents/9417/project/StudentLife_Dataset/Inputs/sensing/phonecharge")
    d_gps = get_GPS("/Users/chris/Documents/9417/project/StudentLife_Dataset/Inputs/sensing/gps")
    # d_wifi = get_wifi("/Users/chris/Documents/9417/project/StudentLife_Dataset/Inputs/sensing/wifi")
    # d_wifi_location = get_wifi_location("/Users/chris/Documents/9417/project/StudentLife_Dataset/Inputs/sensing/wifi_location")
    # d_get_bluetooth = get_bluetooth("/Users/chris/Documents/9417/project/StudentLife_Dataset/Inputs/sensing/bluetooth")
    flourishing_score = Flourishing_score("/Users/chris/Documents/9417/project/StudentLife_Dataset/Outputs")
    # positive_score, negative_score = panas_score("/Users/joshualiu/Desktop/Rock_and_Roll/StudentLife_Dataset/Outputs")
    exit()
    x_train_flurishing = []
    for i in range(60):
        u = "u{:02d}".format(i)
        if u not in flourishing_score:
            continue;
        fea = d_activity[u] + d_audio[u] + d_conversation[u] + d_dark[u] + d_phone_lock[u] \
              + d_innout[u] + d_get_bluetooth[u] + flourishing_score[u]
        #
        x_train_flurishing.append(fea)
    flourishing_score_spreadsheet = np.array(x_train_flurishing)

    x_train_positive = []
    x_train_negative = []

    for i in range(60):
        u = "u{:02d}".format(i)
        if u not in positive_score:
            continue;

        #
        x_train_positive.append(d_activity[u] + d_audio[u] + d_conversation[u] + d_dark[u] + d_phone_lock[u] \
                                + d_innout[u] + d_get_bluetooth[u] + positive_score[u])

        #
        x_train_negative.append(d_activity[u] + d_audio[u] + d_conversation[u] + d_dark[u] + d_phone_lock[u] \
                                + d_innout[u] + d_get_bluetooth[u] + negative_score[u])

    postiive_score_spreadsheet = np.array(x_train_positive)
    negative_score_spreadsheet = np.array(x_train_negative)



# import pandas as pd
# import os
#
# project_dir = os.path.dirname(os.path.dirname(__file__))
# # print(project_dir)
#
# inputs_path = project_dir + '/StudentLife_Dataset/Inputs/sensing'
# outputs_path = project_dir + '/StudentLife_Dataset/Outputs'
#
# # for root, dirs, files in os.walk(inputs_path):
# #     print("root: ", root)
# #     print("dirs: ", dirs)
# #     print("files: ", files)
# #     print('---------------')
#
#
# #input is inputs_path or output_path, return a dictionary, key is dir name, value is a list of it's all csv files's path
# def get_all_csv_path(path):
#     dirs_and_files = {}
#     count = 0
#     all_dirs = []
#     for root, dirs, files in os.walk(path):
#         if dirs:
#             all_dirs = dirs
#             continue
#         if files:
#             all_csvs_path = []
#             for f in files:
#                 if f.endswith('.csv'):
#                     file_path = os.path.join(root, f)
#                     all_csvs_path.append(file_path)
#             dirs_and_files[all_dirs[count]] = sorted(all_csvs_path)
#             count += 1
#     return dirs_and_files
#
# #collect user ID of pre_scores, comparing with data, remove the user data which does not exit in pre.
# dirs_and_files = get_all_csv_path(inputs_path)
# print(dirs_and_files)
#
# #
# # def get_dataframe(dir_name, column_name_list):
# #     for element in dirs_and_files[dir_name]:
# #         df = pd_read_csv
#
# #'physical activity' and 'audio'
# # def count_e(dataframe):
# #     all_df = pd.DataFrame()
# #     number_each_inference = []
# #     for i in range(len(dir_name))
# #         df = pd.read_csv(dir_name)
# #     print(df.columns.values)
#
# # dirs_and_files = get_all_csv_path(inputs_path)
# # print(dirs_and_files)
#
# read_csv('/Users/chris/Documents/9417/project/StudentLife_Dataset/Inputs/sensing/activity/activity_u00.csv')

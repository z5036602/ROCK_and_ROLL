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
from sklearn.preprocessing import StandardScaler 
from sklearn.cross_validation import train_test_split
##get acitivity data
def get_activity_data(file_path):
    all_path = os.listdir(file_path)
    activity_data = {}
    fea = [] 
    acitivity_ratio_list = []
    user_name_list = []
    for path in all_path:
        csv_full_path = os.path.join(file_path,path)
        df = pd.read_csv(csv_full_path)
        #print(df)
        user_name = path.split('_')[1].replace('.csv','')
        activity_ratio = sum(df[' activity inference'].gt(0))/len(df[' activity inference'])
        fea = [activity_ratio]
        activity_data[user_name] = fea
        
    return activity_data
##get audio data
def get_audio_data(file_path):
    all_path = os.listdir(file_path)
    audio_feats = {}
    fea = [] 
    #acitivity_ratio_list = []
    #user_name_list = []
    for path in all_path:
        csv_full_path = os.path.join(file_path,path)
        df = pd.read_csv(csv_full_path)
        all_feats= (df[' audio inference'].value_counts())
        all_counts = df[' audio inference'].count()
        #print(all_feats)
        zero_count_ratio = all_feats[0]/all_counts
        one_count_ratio = all_feats[1]/all_counts
        two_count_ratio = all_feats[2]/all_counts
        fea = [zero_count_ratio,one_count_ratio,two_count_ratio]
        
        #print(all_feats[2])
        user_name = path.split('_')[1].replace('.csv','')
        audio_feats[user_name] = fea;
    return audio_feats;
def get_conversation_data(file_path):
    all_path = os.listdir(file_path)
    conv_feats = {}
    fea = [] 
    #acitivity_ratio_list = []
    #user_name_list = []
    for path in all_path:
        csv_full_path = os.path.join(file_path,path)
        df = pd.read_csv(csv_full_path)
        duration= df[' end_timestamp']-df['start_timestamp']
        total_dur = sum(duration)
        freq_conv = df['start_timestamp'].count()
        #print(sum(duration))
        #print(freq_conv)
        
        fea = [total_dur,freq_conv]
        
        #print(all_feats[2])
        user_name = path.split('_')[1].replace('.csv','')
        #print("user name:", user_name)
        conv_feats[user_name] = fea;
    return conv_feats;
def get_dark(file_path):
    all_path = os.listdir(file_path)
    conv_feats = {}
    fea = [] 
    #acitivity_ratio_list = []
    #user_name_list = []
    for path in all_path:
        csv_full_path = os.path.join(file_path,path)
        df = pd.read_csv(csv_full_path)
        duration= df['end']-df['start']
        total_dur = sum(duration)
        freq_conv = df['end'].count()
        #print(sum(duration))
        #print(freq_conv)
        
        fea = [total_dur,freq_conv]
        
        #print(all_feats[2])
        user_name = path.split('_')[1].replace('.csv','')
        #print("user name:", user_name)
        conv_feats[user_name] = fea;
    return conv_feats;

def get_phone_lock(file_path):
    all_path = os.listdir(file_path)
    conv_feats = {}
    fea = [] 
    #acitivity_ratio_list = []
    #user_name_list = []
    for path in all_path:
        csv_full_path = os.path.join(file_path,path)
        df = pd.read_csv(csv_full_path)
        duration= df['end']-df['start']
        total_dur = sum(duration)
        freq_conv = df['end'].count()
        #print(sum(duration))
        #print(freq_conv)
        
        fea = [total_dur,freq_conv]
        
        #print(all_feats[2])
        user_name = path.split('_')[1].replace('.csv','')
        #print("user name:", user_name)
        conv_feats[user_name] = fea;
    return conv_feats;

def get_bluetooth(file_path):
    all_path = os.listdir(file_path)
    conv_feats = {}
    fea = [] 
    #acitivity_ratio_list = []
    #user_name_list = []
    for path in all_path:
        csv_full_path = os.path.join(file_path,path)
        df = pd.read_csv(csv_full_path)
        
        MAC_encountered = len(df['MAC'].unique())
        #print(sum(duration))
        #print(freq_conv)
        
        fea = [MAC_encountered]
        
        #print(all_feats[2])
        user_name = path.split('_')[1].replace('.csv','')
        #print("user name:", user_name)
        conv_feats[user_name] = fea;
    return conv_feats;

        
def get_in_door_outdoor(file_path):
    all_path = os.listdir(file_path)
    conv_feats = {}
    fea = [] 
    #acitivity_ratio_list = []
    #user_name_list = []
    for path in all_path:
        csv_full_path = os.path.join(file_path,path)
        df = pd.read_csv(csv_full_path)
        total_length = df['time'].count()
        
        count_in = 0
        count_near = 0

        for item in df['time']:
            if('in[' in item):
                count_in+=1
            if('near[' in item):
                count_near+=1
                #if df['time']
  
        count_in_ratio = count_in/total_length
        count_near_ratio = count_near/total_length
        fea = [count_in_ratio,count_near_ratio]
        
        user_name = path.split('_')[2].replace('.csv','')
        #print("user name:", user_name)
        conv_feats[user_name] = fea;
        
    return conv_feats


##flourishing score: $post-pre%
def Flourishing_score(file_path):
    csv_full_path = os.path.join(file_path,'FlourishingScale.csv')
    post_flourishing_score_dictionary = {}
    pre_flourishing_score_dictionary = {}
    flourishing_score_dictionary = []
   
    my_list = []
    
    df = pd.read_csv(csv_full_path)
    numpy_matrix = df.fillna(df.mean()).to_numpy()
    list_flourishing_score = np.sum(numpy_matrix[:,2:9], 1)
    post_index = np.where(numpy_matrix[:,1] == 'post')
    pre_index = np.where(numpy_matrix[:,1] == 'pre')
    flourishing_score_change = {}
    for index in np.nditer(post_index):
        key = numpy_matrix[index,0]
        post_flourishing_score_dictionary [key] = list_flourishing_score[index]
        #print('key_is:',key)
    
    for index in np.nditer(pre_index):
        key = numpy_matrix[index,0]
        pre_flourishing_score_dictionary [key] = list_flourishing_score[index]
        #print('key_is:',key)   
    for key in post_flourishing_score_dictionary:
        
#        u="u{:02d}".format(i)
        if key not in pre_flourishing_score_dictionary:
            continue;
        change =  post_flourishing_score_dictionary[key]-pre_flourishing_score_dictionary[key]
        my_list = [change]
        flourishing_score_change[key] = my_list
#        fea=d_activity[u]+d_audio[u]+d_conversation[u]+d_dark[u]+d_phone_lock[u] \
#        +d_innout[u]+d_get_bluetooth[u]
        #print(type(flourishing_score_change[key]))
#        x_train.append(fea)
    #print(type(flourishing_score_change[key]))
        
   
    return flourishing_score_change
##panas score: $post-pre% for postive and negative
def panas_score(file_path):
    csv_full_path = os.path.join(file_path,'panas.csv')
    post_pos_panas_score_dictionary = {}
    pre_pos_panas_score_dictionary = {}
    post_neg_panas_score_dictionary = {}
    pre_neg_panas_score_dictionary = {}
    panas_positive_score_dictionary = {}
    panas_negative_score_dictionary = {}
    my_list = [] 
    
    df = pd.read_csv(csv_full_path)
    numpy_matrix = df.fillna(df.mean()).to_numpy()
    list_positive_score = np.sum(numpy_matrix[:,[2,5,9,10,12,13,15,16,18]], 1)
    list_negative_score= np.sum(numpy_matrix[:,[3,4,6,7,8,11,14,17,19]], 1)
    post_index = np.where(numpy_matrix[:,1] == 'post')
    pre_index = np.where(numpy_matrix[:,1] == 'pre')
    #print(pre_index,post_index)
    for index in np.nditer(post_index):
        key = numpy_matrix[index,0]
        post_pos_panas_score_dictionary [key] = list_positive_score[index]    
        post_neg_panas_score_dictionary [key] = list_negative_score[index]
    for index in np.nditer(pre_index):
        key = numpy_matrix[index,0]
        pre_pos_panas_score_dictionary [key] = list_positive_score[index]
        pre_neg_panas_score_dictionary [key] = list_negative_score[index]

    for key in post_pos_panas_score_dictionary:
        if key not in pre_pos_panas_score_dictionary:
            continue;
        pos_change = post_pos_panas_score_dictionary[key]-pre_pos_panas_score_dictionary[key]
        neg_change = post_neg_panas_score_dictionary [key]- pre_neg_panas_score_dictionary [key]
        my_list = [pos_change] 
        panas_positive_score_dictionary[key] = my_list
        my_list = [neg_change]
        panas_negative_score_dictionary[key] = my_list 
   
    
##        
##        x_train.append(fea)

        
   
    return panas_positive_score_dictionary,panas_negative_score_dictionary

        
if __name__ == "__main__": 
    
    #print(Flourishing_score("/Users/joshualiu/Desktop/GroupProject/StudentLife_Dataset/Outputs"))
    #print(panas_score("/Users/joshualiu/Desktop/GroupProject/StudentLife_Dataset/Outputs"))
    d_activity = get_activity_data("/Users/joshualiu/Desktop/Rock_and_Roll/StudentLife_Dataset/Inputs/sensing/activity")
    d_audio = get_audio_data("/Users/joshualiu/Desktop/Rock_and_Roll/StudentLife_Dataset/Inputs/sensing/audio")
    d_conversation = get_conversation_data("/Users/joshualiu/Desktop/Rock_and_Roll/StudentLife_Dataset/Inputs/sensing/conversation")
    d_dark = get_dark("/Users/joshualiu/Desktop/Rock_and_Roll/StudentLife_Dataset/Inputs/sensing/dark")
    d_phone_lock = get_phone_lock("/Users/joshualiu/Desktop/Rock_and_Roll/StudentLife_Dataset/Inputs/sensing/phonelock")
    d_innout = get_in_door_outdoor("/Users/joshualiu/Desktop/Rock_and_Roll/StudentLife_Dataset/Inputs/sensing/wifi_location")
    d_get_bluetooth = get_bluetooth("/Users/joshualiu/Desktop/Rock_and_Roll/StudentLife_Dataset/Inputs/sensing/bluetooth")
    flourishing_score = Flourishing_score("/Users/joshualiu/Desktop/Rock_and_Roll/StudentLife_Dataset/Outputs")
    positive_score,negative_score = panas_score("/Users/joshualiu/Desktop/Rock_and_Roll/StudentLife_Dataset/Outputs")
    
    flourishing_score_list = list(flourishing_score.values())
    positive_score_list = list(positive_score.values())
    negative_score_list = list(negative_score.values())
    
    
    
    
    flourishing_median = np.median(flourishing_score_list)
    positive_score_median = np.median(positive_score_list)
    negative_score_median = np.median(negative_score_list)
    y_train_flourishing = []
    y_train_neg_score = []
    y_train_pos_score = []
    
   
    
    for value in flourishing_score.values():
        if value >flourishing_median:
            y_train_flourishing.append(1)
        else:
            y_train_flourishing.append(0)
            
    for value in positive_score.values():
        if value >positive_score_median:
            y_train_pos_score.append(1)
        else:
            y_train_pos_score.append(0)
            
    for value in negative_score.values():
        if value >negative_score_median:
            y_train_neg_score.append(1)
        else:
            y_train_neg_score.append(0)
    
    x_train_flurishing = []
    for i in range(60):
        u="u{:02d}".format(i)
        if u not in flourishing_score:
            continue;
        fea=d_activity[u]+d_audio[u]+d_conversation[u]+d_dark[u]+d_phone_lock[u] \
        +d_innout[u]+d_get_bluetooth[u]
#        
        x_train_flurishing.append(fea)
    x_train_flurishing = np.array(x_train_flurishing)
    
    x_train_positive = []
    x_train_negative = []

    for i in range(60):
        u="u{:02d}".format(i)
        if u not in positive_score:
            continue;
        
#        
        x_train_positive.append(d_activity[u]+d_audio[u]+d_conversation[u]+d_dark[u]+d_phone_lock[u] \
        +d_innout[u]+d_get_bluetooth[u]+positive_score[u])
        
        
#        
        x_train_negative.append(d_activity[u]+d_audio[u]+d_conversation[u]+d_dark[u]+d_phone_lock[u] \
        +d_innout[u]+d_get_bluetooth[u]+negative_score[u])
        
    x_train_positive = np.array(x_train_positive)
    x_train_negative = np.array(x_train_negative)
    
    
    
    #######normalization 
    scaler = StandardScaler()
    x_train_positive=scaler.fit_transform(x_train_positive)
    x_train_negative=scaler.fit_transform(x_train_negative)
    x_train_flurishing=scaler.fit_transform(x_train_flurishing)
    train_test_split()

    
    
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 00:08:15 2019

@author: joshualiu
"""
import pandas as pd 
import os
import statistics as stats

def flouring_score(file_path):
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
        activity_data[user_name] = activity_ratio
    return activity_data





        
if __name__ == "__main__": 
    #print(get_activity_data("/Users/joshualiu/Desktop/GroupProject/StudentLife_Dataset/Inputs/sensing/activity"))
    print(flouring_score("/Users/joshualiu/Desktop/GroupProject/StudentLife_Dataset/Outputs"))
    
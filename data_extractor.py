import os
import pandas as pd
import numpy as np

dataset_path = os.getcwd()

def get_flourishing_score(dataset_path):
    csv_full_path = os.path.join(dataset_path, 'StudentLife_Dataset/Outputs/FlourishingScale.csv')
    post_flourishing_score_dictionary = {}
    pre_flourishing_score_dictionary = {}
    
    df = pd.read_csv(csv_full_path)
    numpy_matrix = df.dropna().to_numpy()
    list_flourishing_score = np.sum(numpy_matrix[:,2:10], 1)
    post_index = np.where(numpy_matrix[:,1] == 'post')
    pre_index = np.where(numpy_matrix[:,1] == 'pre')
    flourishing_score_change = {}
    for index in np.nditer(post_index):
        key = numpy_matrix[index,0]
        post_flourishing_score_dictionary[key] = list_flourishing_score[index]
    
    for index in np.nditer(pre_index):
        key = numpy_matrix[index,0]
        pre_flourishing_score_dictionary[key] = list_flourishing_score[index]
    
    print('Post Entries Count : ', len(post_flourishing_score_dictionary))
    print('Pre Entries Count : ', len(pre_flourishing_score_dictionary))

    flourishing_score_diff = {}  
    for key in post_flourishing_score_dictionary:
        if key in pre_flourishing_score_dictionary:
            flourishing_score_diff[key] = post_flourishing_score_dictionary[key] - pre_flourishing_score_dictionary[key]

    print('Aggregates Count :', len(flourishing_score_diff))
    return flourishing_score_diff

def get_panas_score(dataset_path):
    csv_full_path = os.path.join(dataset_path, 'StudentLife_Dataset/Outputs/panas.csv')    
    post_pos_panas_score_dictionary = {}
    pre_pos_panas_score_dictionary = {}
    post_neg_panas_score_dictionary = {}
    pre_neg_panas_score_dictionary = {}
    panas_positive_diff = {}
    panas_negative_diff = {}
    panas_positive_div = {}
    panas_negative_div = {}
    my_list = [] 
    
    df = pd.read_csv(csv_full_path)
    numpy_matrix = df.dropna().to_numpy()
    list_positive_score = np.sum(numpy_matrix[:, [2, 5, 9, 10, 12, 13, 15, 16, 18]], 1)
    list_negative_score= np.sum(numpy_matrix[:, [3, 4, 6, 7, 8, 11, 14, 17, 19]], 1)
    post_index = np.where(numpy_matrix[:,1] == 'post')
    pre_index = np.where(numpy_matrix[:,1] == 'pre')

    for index in np.nditer(post_index):
        key = numpy_matrix[index,0]
        post_pos_panas_score_dictionary[key] = list_positive_score[index]    
        post_neg_panas_score_dictionary[key] = list_negative_score[index]
    for index in np.nditer(pre_index):
        key = numpy_matrix[index,0]
        pre_pos_panas_score_dictionary[key] = list_positive_score[index]
        pre_neg_panas_score_dictionary[key] = list_negative_score[index]

    print('Post Entries Count : ', len(post_pos_panas_score_dictionary))
    print('Pre Entries Count : ', len(pre_pos_panas_score_dictionary))

    for key in post_pos_panas_score_dictionary:
        if key in pre_pos_panas_score_dictionary:
            panas_positive_diff[key] = post_pos_panas_score_dictionary[key] - pre_pos_panas_score_dictionary[key]
            panas_negative_diff[key] = post_neg_panas_score_dictionary[key] - pre_neg_panas_score_dictionary[key]

    print('Aggregates Count :', len(panas_positive_diff))
    return panas_positive_diff, panas_negative_diff


def get_activity_data(file_path):
    all_path = os.listdir(file_path)
    activity_data = {}
    fea = [] 
    for path in all_path:
        csv_full_path = os.path.join(file_path,path)
        df = pd.read_csv(csv_full_path)
        user_name = path.split('_')[1].replace('.csv','')
        activity_ratio = sum(df[' activity inference'].gt(0))/len(df[' activity inference'])
        fea = [activity_ratio]
        activity_data[user_name] = fea
    return activity_data, ['activity_ratio']

def get_activity_time(file_path):
    all_path = os.listdir(file_path)
    activity_data = {}
    feature_names = ['total_rest_sec', 'total_walking_sec', 'total_running_sec', 'total_unknown_sec']
    for path in all_path:
        csv_full_path = os.path.join(file_path, path)
        df = pd.read_csv(csv_full_path)
        numpy_matrix = df.dropna().to_numpy()
        total_activity_time = [0, 0, 0, 0]
        last_activity_initial_time = numpy_matrix[0][0]
        last_activity = numpy_matrix[0][1]
        for i in range(1, numpy_matrix.shape[0]):
            if numpy_matrix[i, 1] != last_activity:
                total_activity_time[int(last_activity)] += (numpy_matrix[i, 0] - last_activity_initial_time)
                last_activity_initial_time = numpy_matrix[i, 0]        
                last_activity = numpy_matrix[i, 1]
        total_activity_time[int(last_activity)] += (numpy_matrix[i-1, 0] - last_activity_initial_time)
        user_name = path.split('_')[1].replace('.csv','')
        activity_data[user_name] = total_activity_time
    return activity_data, feature_names

def get_audio_data(file_path):
    all_path = os.listdir(file_path)
    audio_feats = {}
    fea = [] 
    feature_names = ['zero_count_ratio', 'one_count_ratio', 'two_count_ratio']
    for path in all_path:
        csv_full_path = os.path.join(file_path,path)
        df = pd.read_csv(csv_full_path)
        all_feats= (df[' audio inference'].value_counts())
        all_counts = df[' audio inference'].count()
        zero_count_ratio = all_feats[0]/all_counts
        one_count_ratio = all_feats[1]/all_counts
        two_count_ratio = all_feats[2]/all_counts
        fea = [zero_count_ratio,one_count_ratio,two_count_ratio]
        user_name = path.split('_')[1].replace('.csv','')
        audio_feats[user_name] = fea;
    return audio_feats, feature_names

def get_audio_time(file_path):
    all_path = os.listdir(file_path)
    audio_type_data = {}
    #found that value of 3 - i.e. unknown is not seen for audio
    feature_names = ['total_silent_sec', 'total_voice_sec', 'total_noise_sec']
    for path in all_path:
        csv_full_path = os.path.join(file_path, path)
        df = pd.read_csv(csv_full_path)
        numpy_matrix = df.dropna().to_numpy()
        total_audio_type_time = [0, 0, 0]
        last_audio_type_initial_time = numpy_matrix[0][0]
        last_audio_type = numpy_matrix[0][1]
        for i in range(1, numpy_matrix.shape[0]):
            if numpy_matrix[i, 1] != last_audio_type:
                total_audio_type_time[int(last_audio_type)] += (numpy_matrix[i, 0] - last_audio_type_initial_time)
                last_audio_type_initial_time = numpy_matrix[i, 0]        
                last_audio_type = numpy_matrix[i, 1]
        total_audio_type_time[int(last_audio_type)] += (numpy_matrix[i-1, 0] - last_audio_type_initial_time)
        user_name = path.split('_')[1].replace('.csv','')
        audio_type_data[user_name] = total_audio_type_time
    return audio_type_data, feature_names


def get_end_minus_start_data(file_path):
    all_path = os.listdir(file_path)
    diff_data = {}
    for path in all_path:
        csv_full_path = os.path.join(file_path, path)
        df = pd.read_csv(csv_full_path)
        numpy_matrix = df.dropna().to_numpy()
        diff_time = sum(numpy_matrix[:, 1] - numpy_matrix[:, 0])
        user_name = path.split('_')[1].replace('.csv','')
        diff_data[user_name] = [diff_time]
    return diff_data

if __name__ == '__main__':
    dataset_path = os.getcwd()
    input_data_path = os.path.join(dataset_path, 'StudentLife_Dataset/Inputs/sensing')

    activity_data = get_activity_data(os.path.join(input_data_path, 'activity/'))
    print(activity_data)

    activity_time, act_features = get_activity_time(os.path.join(input_data_path, 'activity/'))    
    print(activity_time)
    print(type(activity_time))
    print(act_features)  

    audio_time, aud_features = get_audio_time(os.path.join(input_data_path, 'audio/'))
    print(audio_time)

    conversation_time = get_end_minus_start_data(os.path.join(input_data_path, 'conversation/'))
    print(conversation_time)

    dark_time = get_end_minus_start_data(os.path.join(input_data_path, 'dark/'))
    print(conversation_time)

    phcharge_time = get_end_minus_start_data(os.path.join(input_data_path, 'phonecharge/'))
    print(phcharge_time)

    phonelock_time = get_end_minus_start_data(os.path.join(input_data_path, 'phonelock/'))
    print(phonelock_time)
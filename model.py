import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

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
    flourishing_score_div = {}    
    for key in post_flourishing_score_dictionary:
        if key in pre_flourishing_score_dictionary:
        	flourishing_score_diff[key] = post_flourishing_score_dictionary[key] - pre_flourishing_score_dictionary[key]
        	flourishing_score_div[key] = post_flourishing_score_dictionary[key] / pre_flourishing_score_dictionary[key]

    print('Aggregates Count :', len(flourishing_score_div))
    return flourishing_score_diff, flourishing_score_div

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

            panas_positive_div[key] = post_pos_panas_score_dictionary[key] / pre_pos_panas_score_dictionary[key]
            panas_negative_div[key] = post_neg_panas_score_dictionary[key] / pre_neg_panas_score_dictionary[key]

    print('Aggregates Count :', len(panas_positive_diff))
    return panas_positive_diff, panas_negative_diff, panas_positive_div, panas_negative_div


def get_activity_data(file_path):
    all_path = os.listdir(file_path)
    activity_data = {}
    data = {}
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
                total_activity_time[int(last_activity)] += (numpy_matrix[i-1, 0] - last_activity_initial_time)
                last_activity_initial_time = numpy_matrix[i, 0]        
                last_activity = numpy_matrix[i, 1]
        total_activity_time[int(last_activity)] += (numpy_matrix[i-1, 0] - last_activity_initial_time)

        user_name = path.split('_')[1].replace('.csv','')
        data[user_name] = total_activity_time
    activity_data['data'] = data
    activity_data['feature_names'] = feature_names
    return activity_data


def get_audio_data(file_path):
    all_path = os.listdir(file_path)
    audio_type_data = {}
    data = {}
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
                total_audio_type_time[int(last_audio_type)] += (numpy_matrix[i-1, 0] - last_audio_type_initial_time)
                last_audio_type_initial_time = numpy_matrix[i, 0]        
                last_audio_type = numpy_matrix[i, 1]
        total_audio_type_time[int(last_audio_type)] += (numpy_matrix[i-1, 0] - last_audio_type_initial_time)

        user_name = path.split('_')[1].replace('.csv','')
        data[user_name] = total_audio_type_time
    audio_type_data['data'] = data
    audio_type_data['feature_names'] = feature_names
    return audio_type_data


def get_end_minus_start_data(file_path, feature_name):
    all_path = os.listdir(file_path)
    all_diff_data = {}
    data = {}
    for path in all_path:
        csv_full_path = os.path.join(file_path, path)
        df = pd.read_csv(csv_full_path)
        numpy_matrix = df.dropna().to_numpy()
        diff_time = sum(numpy_matrix[:, 1] - numpy_matrix[:, 0])
        user_name = path.split('_')[1].replace('.csv','')
        data[user_name] = [diff_time]
    all_diff_data['data'] = data
    all_diff_data['feature_names'] = [feature_name]
    return all_diff_data


def add_feature_to_dataframe(df, feature_info):
    """
        adds all features from feature_info into df

        Arguments :
        df : existing dataframe containing labels and other features
        feature_info : sample given below
            {
                'data': {
                    'u00': [4507713.0, 369988.0, 89425.0, 169967.0],
                    'u07': [3852787, 150096, 51295, 76300]
                }
                'feature_names': ['total_rest_time', 'total_walking_time', 'total_running_time', 'total_unknown_time']
            }

            elements of the list for each userids are values for features mentioned in 'feature_names'
    """
    user_ids = df.index.values
    feature_data = feature_info['data']
    feature_names = feature_info['feature_names']
    processed_feature_data = [[] for i in range(len(feature_names))]
    for uid in user_ids:
        if uid in feature_data:
            for i in range(len(feature_names)):
                processed_feature_data[i].append(feature_data[uid][i])

    for i in range(len(feature_names)):
        df[feature_names[i]] = processed_feature_data[i] 
    return None

def add_input_features(df):
    input_data_path = os.path.join(dataset_path, 'StudentLife_Dataset/Inputs/sensing')

    add_feature_to_dataframe(df, get_activity_data(os.path.join(input_data_path, 'activity/')))    
    add_feature_to_dataframe(df, get_audio_data(os.path.join(input_data_path, 'audio/')))  
    add_feature_to_dataframe(df, get_end_minus_start_data(os.path.join(input_data_path, 'conversation/'), 'total_conversation_sec'))  
    add_feature_to_dataframe(df, get_end_minus_start_data(os.path.join(input_data_path, 'dark/'), 'total_dark_time_sec'))
    add_feature_to_dataframe(df, get_end_minus_start_data(os.path.join(input_data_path, 'phonecharge/'), 'total_phcharge_sec'))
    add_feature_to_dataframe(df, get_end_minus_start_data(os.path.join(input_data_path, 'phonelock/'), 'total_phlock_sec'))

    return None

def normalize_feature(df, feature_name):
    df[feature_name] = (df[feature_name] - min(df[feature_name])) / (max(df[feature_name]) - min(df[feature_name]))
    return None

def convert_to_binary_label(df, feature_name):
    median = df[feature_name].median()
    df.loc[df[feature_name] < median, feature_name] = 0
    df.loc[df[feature_name] >= median, feature_name] = 1
    return None

def get_features_and_labels(df):
    feature_data = df.iloc[:, 1:]
    label_data = df.iloc[:, 0]
    return feature_data, label_data    

def predict_flourishing_score():
    processed_data_file_name = 'flourishing_score_data.csv'

    if not os.path.exists(processed_data_file_name):
        flourishing_score_diff, flourishing_score_div = get_flourishing_score(dataset_path)
        #taking only flourishing_score_diff for now
        df = pd.DataFrame.from_dict(flourishing_score_diff, orient='index')
        df.columns = ['fl_score']
        add_input_features(df)
        df.to_csv(processed_data_file_name, header=True)
    else:
        print('did not read original files')
        df = pd.read_csv(processed_data_file_name, index_col = 0)

    convert_to_binary_label(df, 'fl_score')

    for feature in df.columns[1:]:
        print('normalizing ', feature)
        normalize_feature(df, feature)

    training_data = df[:27]
    test_data = df[27:]
    
    train_feature_data, train_label_data = get_features_and_labels(training_data)
    test_feature_data, test_label_data = get_features_and_labels(test_data)

    print(train_feature_data.describe())
    print(train_label_data.describe())

    model = LogisticRegression(solver = 'liblinear')
    model.fit(train_feature_data, train_label_data)

    train_pred = model.predict(train_feature_data)
    train_pred_prob = np.array([val[1] for val in model.predict_proba(train_feature_data)])  
    test_pred = model.predict(test_feature_data)
    test_pred_prob = np.array([val[1] for val in model.predict_proba(test_feature_data)])

    get_metric_scores(train_label_data, train_pred_prob, train_pred, test_label_data, test_pred_prob, test_pred)


def get_metric_scores(train_label_data, train_pred_prob, train_pred, test_label_data, test_pred_prob, test_pred):
    auc = roc_auc_score(train_label_data, train_pred_prob)
    print('Train AUC : ', auc)
    auc = roc_auc_score(test_label_data, test_pred_prob)
    print('Test AUC : ', auc)

    accuracy = accuracy_score(train_label_data, train_pred)
    print('Train accuracy : ', accuracy)
    accuracy = accuracy_score(test_label_data, test_pred)
    print('Test accuracy : ', accuracy)


if __name__ == '__main__':
    predict_flourishing_score()



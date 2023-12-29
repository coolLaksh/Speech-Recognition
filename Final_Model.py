#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import librosa
import os
from collections import Counter
import librosa
import numpy as np


# In[2]:


direrctory_train = r""
direrctory_test = r""


# In[3]:


def load_train_Data(path):

    data_dict= {}

    for subdir in os.listdir(path):

        data_dict[subdir] = []

        subdir_path = os.path.join(path,subdir)
        if os.path.isdir(subdir_path) and subdir != "_background_noise_":
            for root, _, files in os.walk(subdir_path):
                for file in files:
                    if file.endswith(".wav"):
                        
                        audio_file_path = os.path.join(root, file)
                        y, sr = librosa.load(audio_file_path)

                        data_dict[subdir].append([y,sr])

    key_lst = []
    signal_lst = []
    sr = []

    for key,value in data_dict.items():
        for item in value:
            key_lst.append(key)
            signal_lst.append(item[0])
            sr.append(item[1])

    helper_dict = {"Labels" : key_lst, "signal" : signal_lst, "SR":sr}

    dataset = pd.DataFrame(helper_dict)
    
    return dataset


def extract_audio_features(file_path):
    signal, sr = librosa.load(file_path)

    return {
        'File': os.path.basename(file_path),
        'signal': signal,
        'SR': sr
    }

def load_test_Data(directory):
    
    audio_data = []
    
    for filename in os.listdir(directory):
        if filename.endswith(".wav"): 
            file_path = os.path.join(directory, filename)
            audio_info = extract_audio_features(file_path)
            audio_data.append(audio_info)

    df = pd.DataFrame(audio_data)
    return df


# In[4]:


train_Data = load_train_Data(direrctory_train)


# In[5]:


def convert_labels_to_numericals(labels):
    label_mapping = {'right': 0, 'eight': 1, 'cat': 2, 'tree': 3, 'bed': 4,
        'happy': 5, 'go': 6, 'dog': 7, 'no': 8, 'wow': 9,
        'nine': 10, 'left': 11, 'stop': 12, 'three': 13, 'sheila': 14,
        'one': 15, 'bird': 16, 'zero': 17, 'seven': 18, 'up': 19,
        'marvin': 20, 'two': 21, 'house': 22, 'down': 23, 'six': 24,
        'yes': 25, 'on': 26, 'five': 27, 'off': 28, 'four': 29}

    return [label_mapping[label] for label in labels]


# In[6]:


train_Data["Labels"] = convert_labels_to_numericals(list(train_Data["Labels"]))


# In[7]:


train_Data


# In[8]:


from sklearn.model_selection import train_test_split

training_Data, testing_Data, train_labels, test_labels = train_test_split(
    train_Data.drop(columns=["Labels"]), train_Data["Labels"], shuffle=True,
    stratify=train_Data["Labels"],test_size=0.10
)


# In[9]:


training_Data["Labels"] = train_labels
training_Data


# In[10]:


import numpy as np
from sklearn import preprocessing
import numpy as np
from sklearn.preprocessing import MinMaxScaler
# def calculate_delta(array):
#     """Calculate and return the second derivative of the given feature vector matrix"""

#     rows, cols = array.shape
#     deltas1 = np.zeros((rows, cols))
#     deltas2 = np.zeros((rows, cols))

#     N = 2

#     for i in range(rows):
#         index = []
#         j = 1

#         while j <= N:
#             if i - j < 0:
#                 first = 0
#             else:
#                 first = i - j
#             if i + j > rows - 1:
#                 second = rows - 1
#             else:
#                 second = i + j
#             index.append((second, first))
#             j += 1

        
#         first_derivative = (
#             array[index[0][0]] - array[index[0][1]]
#             + 2 * (array[index[1][0]] - array[index[1][1]])
#         ) / 10

#         second_derivative = (
#             array[index[0][0]] - 2 * array[i] + array[index[0][1]]
#             + 2 * (array[i] - array[index[1][0]] + array[index[1][1]])
#         ) / 10

#         deltas1[i] = first_derivative
#         deltas2[i] = second_derivative

#     return deltas1, deltas2

def extract_features(audio,rate):

    mfcc_feature = librosa.feature.mfcc(y=audio, sr=rate, n_mfcc=20).T
    delta1 = np.gradient(mfcc_feature, axis=0)
    delta2 = np.gradient(delta1,axis=0)
    
    scaler = MinMaxScaler(feature_range=(0,1))

    mfccs_scaled = scaler.fit_transform(mfcc_feature)
    gradients_scaled = scaler.fit_transform(delta1)
    double_gradients_scaled = scaler.fit_transform(delta2)


    combined = np.hstack((mfccs_scaled,gradients_scaled,double_gradients_scaled))
    
    return combined


# In[11]:


def features_Extraction_Formatting(group_data):

    ret_dict = {}

    for label, group in group_data:

        features = np.asarray(())

        for index,DataFrame in group.iterrows():

            DataFrame = DataFrame.drop(columns=["Labels"])
            try:
                vector_feature = extract_features(DataFrame["signal"],DataFrame["SR"])
            except:
                continue
            if features.size == 0:
                features = vector_feature
            else:
                features = np.vstack((features, vector_feature))

        ret_dict[label] = features

    return ret_dict


# In[ ]:





# In[12]:


training_data_valuable = training_Data
grouped_data = training_data_valuable.groupby("Labels")


# In[13]:


extracted_features = features_Extraction_Formatting(grouped_data)


# In[14]:


from sklearn.preprocessing import MinMaxScaler

def Normalize_Data(data_dict):

    normalize_dict = {}
    normalize_data = {}

    for key, data in data_dict.items():

        scaler = MinMaxScaler()
        normalized_Data = scaler.fit_transform(data)

        normalize_dict[key] = scaler
        normalize_data[key] = normalized_Data

    return normalize_dict, normalize_data


# In[ ]:





# In[ ]:





# In[ ]:





# In[15]:


from sklearn.mixture import GaussianMixture

def train_GMMs(data_dict):

    ret_dict = {}

    for key,data in data_dict.items():
        gmm = GaussianMixture(n_components=10,covariance_type='full',n_init=3)
        gmm.fit(data)

        ret_dict[key] = gmm

    return ret_dict


# In[16]:


models_dict = train_GMMs(extracted_features)


# In[17]:


def make_Predictions(data_point, sr,data_dict):

    likelihoods = {}

    data_point = extract_features(data_point, sr)

    for key, model in data_dict.items():

        scores = np.array(model.score(data_point))
        likelihoods[key] = scores
    
    max_key = max(likelihoods, key=lambda k: likelihoods[k])
    return max_key

def predict(data_s, rates,model_dict):

    ret_Arr = []

    for signal, rate in zip(data_s, rates):
        prediction = make_Predictions(signal,rate, model_dict)
        ret_Arr.append(prediction)
    return ret_Arr


# In[18]:


predictions = predict(list(testing_Data["signal"]), list(testing_Data["SR"]),models_dict)


# In[19]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

accuracy = accuracy_score(list(test_labels), predictions)
precision = precision_score(list(test_labels), predictions, average='weighted')
recall = recall_score(list(test_labels), predictions, average='weighted')
f1 = f1_score(list(test_labels), predictions, average='weighted')

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)


# In[ ]:





# In[20]:


test_data = load_test_Data(direrctory_test)


# In[21]:


test_Labels = predict(list(test_data["signal"]), list(test_data["SR"]), models_dict)


# In[22]:


test_data["Label"] = test_Labels


# In[23]:


test_frame = pd.read_csv("KAGGLE-2/test.csv")

IDs = list(test_frame["ID"])
names = list(test_frame["AUDIO_FILE"])

result = {}

for ID, name in zip(IDs, names):
    result_row = test_data[test_data['File'] == name][['File', 'Label']]
    result[ID] = list(result_row["Label"])[0]


# In[24]:


testing = pd.DataFrame({"ID" : result.keys(), "TARGET" : result.values()})


# In[25]:


testing.to_csv("IETG.csv",index=False)


# In[26]:


testing


# In[ ]:





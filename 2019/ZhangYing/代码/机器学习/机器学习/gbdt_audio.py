# -*- coding: utf-8 -*-
# @File    : train_xgb.py
# @Software: PyCharm

import numpy as np
import gbdt_model as gbdt

# train_feature = np.load("./feature_data/train_feature.npy")
# train_label = np.load("./feature_data/train_label.npy")
# test_feature = np.load("./feature_data/test_feature.npy")
# test_label = np.load("./feature_data/test_label.npy")
model_path = "./gbdt_model_cv.pkl"


# dao ru feature
import pickle
import numpy as np
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, train_test_split

import os
import sklearn
import pickle
import numpy as np
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold,train_test_split


# features_path = "/media/lab307/aa9d6aa6-56d4-49e0-80cd-908035f93743/data/cctv120170303/features/"
#
#
# def get_file_list(Path):
#     names = []
#     for filename in os.listdir(Path):
#         name = os.path.join(Path, filename)
#         names.append(name)
#     return sorted(names)
#
#
# files = get_file_list(features_path)


f = open("features-audio-not.pkl", "rb")
features = pickle.load(f)
f.close()
print(len(features))
f = open("features-audio-ad.pkl", "rb")
ads = pickle.load(f)
print(len(ads))
features.extend(ads)
f.close()
labels = [0.0]*(len(features)-len(ads))
labels.extend([1.0]*len(ads))
features_mat = []

for item in features:
    features_mat.extend(item)

pca = PCA(n_components=84)
features_84 = pca.fit_transform(features_mat)


def mat_to_vec(features):
    features_vec = []
    for i in range(len(labels)):
        # concatenate 4 feature vectors as one
        # features_vec.append(np.concatenate(features_300[4*i:4*i+4]))

        # mean 4 feature vectors as one
        features_vec.append(np.mean(features[4 * i:4 * i + 4], axis=0))

        # sum 4 feature vectors as one
        # features_vec.append(np.sum(features_300[4*i:4*i+4],axis=0))
    return features_vec


# features_vec = mat_to_vec(features_84)
norm = preprocessing.normalize(features_84)

train_feature,test_feature,train_label,test_label = train_test_split(features_84, labels, test_size=0.25)

train_label=np.array(train_label)
test_label =np.array(test_label)

#训练
gbdt.train_gbdt_cv(train_feature, train_label, test_feature, test_label, model_path)
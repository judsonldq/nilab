# -*- coding: utf-8 -*-
# @File    : train_xgb.py
# @Software: PyCharm

import numpy as np
import rf_model as rf

# train_feature = np.load("./feature_data/train_feature.npy")
# train_label = np.load("./feature_data/train_label.npy")
# test_feature = np.load("./feature_data/test_feature.npy")
# test_label = np.load("./feature_data/test_label.npy")
model_path = "./rf_model_cv.pkl"



# dao ru feature
import pickle
import numpy as np
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, train_test_split

# dao ru feature
import pickle
import numpy as np
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, train_test_split

f = open("features-image-not.pkl", "rb")
image = pickle.load(f)
f.close()
f = open("features-image-ad.pkl", "rb")
image_ad = pickle.load(f)
f.close()
image.extend(image_ad)
labels = [0.0]*(len(image)-len(image_ad))
labels.extend([1.0]*len(image_ad))
image_mat = []
for item in image:
    if len(item) != 4:
        print("Error")
    image_mat.extend(item)
pca = PCA(n_components=84)
image_84 = pca.fit_transform(image_mat)


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


image_vec = mat_to_vec(image_84)
image_vec = np.array(image_vec)

norm = preprocessing.normalize(image_vec)

train_feature,test_feature,train_label,test_label = train_test_split(image_vec, labels, test_size=0.25)

train_label=np.array(train_label)
test_label =np.array(test_label)

# print(np.sum(train_label))

# print(len(train_label))

#训练
rf.train_rf_cv(train_feature, train_label, test_feature, test_label, model_path)
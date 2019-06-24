# -*- coding: utf-8 -*-
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


#f = open("features-audio-not.pkl", "rb")
#features = pickle.load(f)
#f.close()

with open("features-audio-not.pkl", "rb") as f:
    features = pickle.load(f, encoding='bytes')
print(len(features))

#f = open("features-audio-ad.pkl", "rb")
#ads = pickle.load(f)
#f.close()

with open("features-audio-ad.pkl", "rb") as f1:
    ads = pickle.load(f1, encoding='bytes')
print(len(ads))
features.extend(ads)
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
X_train, X_test, y_train, y_test = train_test_split(features_84, labels, test_size=0.25)
clf = SVC(C=3.5, gamma=0.02, cache_size=1000)
clf.fit(X_train, y_train)
acc = clf.score(X_test, y_test)
print(acc)

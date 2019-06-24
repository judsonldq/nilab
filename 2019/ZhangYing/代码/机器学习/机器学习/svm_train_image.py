# dao ru feature
import pickle
import numpy as np
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, train_test_split

with open("features-image-not.pkl", "rb") as f:
    image = pickle.load(f, encoding = 'bytes')

#f = open("features-image-not.pkl", "rb")
#image = pickle.load(f, encoding = 'bytes')
#f.close()

with open("features-image-ad.pkl", "rb") as f1:
    image_ad = pickle.load(f1, encoding='bytes')

#f = open("features-image-ad.pkl", "rb")
#image_ad = pickle.load(f, encoding = 'bytes')
#f.close()
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
#print(pca.explained_variance_ratio_)

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


X_train, X_test, y_train, y_test = train_test_split(image_vec, labels, test_size=0.25)
clf = SVC(C=3.5, gamma=0.02, cache_size=1000)
clf.fit(X_train, y_train)
acc = clf.score(X_test, y_test)
print(acc)

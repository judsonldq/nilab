# -*- coding: utf-8 -*-
# @File    : train_xgb.py
# @Software: PyCharm
import pickle
import numpy as np
#from sklearn import grid_search
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import accuracy_score

def eval(test_label, predict_label):
    TP = 0
    FN = 0
    FP = 0
    TN = 0

    for i in range(test_label.shape[0]):
        if test_label[i] == 1 and predict_label[i] == 1:
            TP += 1
        elif test_label[i] == 1 and predict_label[i] == 0:
            FN += 1
        elif test_label[i] == 0 and predict_label[i] == 1:
            FP += 1
        else:
            TN += 1

    recall = TP / (TP + FN)
    precision = TP / (TP + FP)
    acc = (TP + TN)/(TP + TN + FP + FN)
    f1_score = 2*precision*recall/(precision+recall)

    return precision, recall, acc, f1_score

def train_gbdt_cv(train_feature, train_label, test_feature, test_label, model_path):
    """
    使用交叉验证选择超参数
    :param train_feature:训练特征
    :param train_label:训练标签
    :param test_feature:测试特征
    :param test_label:测试标签
    :param model_path:模型路径
    :return:
    """
    param_grid = {
        "n_estimators":[400,500,600,700],
        "max_depth":[6,7,8]
    }
    model = GradientBoostingClassifier(
        learning_rate=0.01,
        subsample=0.8,
        min_samples_split=4
    )
    grid_model = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring='accuracy',
        cv=5
    )

    grid_model.fit(train_feature, train_label)
    print("best score is %f" % (grid_model.best_score_))
    print("best param is %s" % (grid_model.best_params_))
    best_model = grid_model.best_estimator_
    predict_pro = best_model.predict_proba(test_feature)[:, 1]
    fpr, tpr, thresholds = roc_curve(test_label, predict_pro, pos_label=2)
    # 计算AUC
    # result_auc = auc(fpr, tpr)
    # print("-" * 20)
    # # print(predict_pro)
    # # print(fpr)
    # # print(tpr)
    # # print(thresholds)
    # print("auc is %f"%(result_auc))
    # print("-" * 20)

    # 预测label
    predict_label = best_model.predict(test_feature)
    # print("=" * 20)
    # print(predict_label)
    # print("=" * 20)

    # precision, recall, acc, f1_score = eval(test_label, predict_label)
    acc = accuracy_score(test_label, predict_label)
    print("*" * 20)
    print("accuracy is %f" % (acc))
    # print("recall is %f" % (recall))
    # print("precision is %f" % (precision))
    # print("f1_score is %f" % (f1_score))
    print("*" * 20)

    with open(model_path, "wb") as f:
        pickle.dump(best_model, f)

def predict_gbdt(user_feature, user_list, submission_name, predict_prob_name):

    # load model
    with open("./saved_model/gbdt_model_cv.pkl", 'rb') as f:
        model = pickle.load(f)

    predict_prob = model.predict_proba(user_feature)[:,1]

    predict_prob_sort = sorted(predict_prob, reverse=True)
    threshold = predict_prob_sort[33000]
    print("threshold is %f"%(threshold))
    result = []
    for i in range(len(predict_prob)):
        if predict_prob[i] >= threshold:
            result.append(user_list[i])

    with open("./result/" + submission_name + ".txt", 'w', encoding='utf-8') as f:
        for i in result:
            f.write(str(i))
            f.write("\n")

    np.save("./result_analysis/" + predict_prob_name, predict_prob)


def predict_gbdt_prob(user_feature):
    # load model
    with open("./saved_model/gbdt_model_cv.pkl", 'rb') as f:
        model = pickle.load(f)

    predict_prob = model.predict_proba(user_feature)[:, 1]
    return predict_prob
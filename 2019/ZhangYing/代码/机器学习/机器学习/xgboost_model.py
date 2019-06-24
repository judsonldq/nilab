# -*- coding: utf-8 -*-
# @File    : xgboost_model.py
# @Software: PyCharm

import pickle
import xgboost as xgb
import numpy as np
from xgboost.sklearn import XGBClassifier
from sklearn import grid_search
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
    f1_score = 2*precision*recall/(precision+recall+0.0000001)

    return precision, recall, acc, f1_score

def train_xgb_cv(train_feature, train_label, test_feature, test_label, model_path):
    """
    使用网格搜索和交叉验证的方法寻找最优超参数
    :param train_feature: 训练集特征向量，类型为np.array，shape为sample_number*feature_dim
    :param train_label: 训练集标签，类型为np.array，shape为sample_number
    :param test_feature: 测试集特征向量，类型为np.array，shape为sample_number*feature_dim
    :param test_label: 测试集标签，类型为np.array，shape为sample_number
    :param model_path: 模型的保存路径
    """
    #用于网格搜索的超参数组合
    param_grid = {
        "max_depth": [3,4, 5,6],
        "learning_rate": [0.001],
        'reg_lambda': [0.3,  0.5],
        'reg_alpha': [0.3, 0.5],
        'subsample': [0.8],
        'min_child_weight': [5],
        'n_estimators':[1700,2000,2200,2500]
    }
    #建立模型
    model = XGBClassifier(objective='binary:logistic', silent=0, nthread=8)
    grid_model = grid_search.GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring='accuracy',
        cv=5
    )
    #训练
    grid_model.fit(train_feature, train_label)
    #打印出最好的训练结果和对应的超参数
    print("best score is %f" % (grid_model.best_score_))
    print("best param is %s" % (grid_model.best_params_))
    #取出最好的模型
    best_model = grid_model.best_estimator_

    #使用模型在测试集上进行概率预测，如果需要计算AUC，可以取消这部分的注释
    # predict_pro = best_model.predict_proba(test_feature)[:, 1]
    # fpr, tpr, thresholds = roc_curve(test_label, predict_pro, pos_label=1)
    # # 计算AUC
    # result_auc = auc(fpr, tpr)
    # print("-" * 20)
    # print(predict_pro)
    # # print(fpr)
    # # print(tpr)
    # # print(thresholds)
    # # print(result_auc)
    # print("-" * 20)

    # 预测label
    predict_label = best_model.predict(test_feature)
    print("=" * 10 + "predict label" + "="*10)
    print(predict_label)
    print("=" * 30)

    #计算精准率，召回率，准确率和F1值
    # precision, recall, acc, f1_score = eval(test_label, predict_label)
    acc = accuracy_score(test_label, predict_label)
    print("*" * 20)
    print("accuracy is %f" % (acc))
    # print("recall is %f" % (recall))
    # print("precision is %f" % (precision))
    # print("f1_score is %f" % (f1_score))
    print("*" * 20)
    #保存模型
    with open(model_path, "wb") as f:
        pickle.dump(best_model, f)

def predict_xgb_prob(user_feature, model_path):
    """
    使用训练好的模型预测概率
    :param user_feature:特征向量
    :param model_path: 模型路径
    :return: 预测得到的概率
    """
    # load model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    predict_prob = model.predict_proba(user_feature)[:, 1]
    return predict_prob

def predict_xgb_label(user_feature, model_path):
    """
    使用训练好的模型预测标签
    :param user_feature:特征向量
    :param model_path: 模型路径
    :return: 预测得到的标签
    """
    # load model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    predict_label = model.predict(user_feature)
    return predict_label
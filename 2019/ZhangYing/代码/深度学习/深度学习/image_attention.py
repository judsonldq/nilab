# -*- coding: utf-8 -*-
import pickle
import numpy  as np
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, train_test_split
import pandas as pd
import numpy as np
import math
import time
import pandas as pd
import numpy as np
import math
import time
import pandas as pd
import numpy as np
import math
import time
from keras import Sequential
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Activation, Dropout
from keras import optimizers
from keras.models import load_model
from keras import backend as K
from sklearn import metrics
from sklearn.metrics import mean_squared_error
#带lstm的attention模型
from keras.layers import merge
from keras.layers.core import *
from keras.layers.recurrent import LSTM
from keras.models import *
# from attention_utils import get_activations, get_data_recurrent
from keras.layers import Multiply
"""
输入广告／非广告特征向量的pickle文件，进行特征融合
将融合的特征使用SVM进行训练并保存
"""

f = open("features-image-not.pkl", "rb")
image = pickle.load(f)
f.close()
f = open("features-image-ad.pkl", "rb")
image_ad = pickle.load(f)
f.close()
image.extend(image_ad)

labels = [0.0]*(len(image)-len(image_ad))# 根据广告非广告的数据长度分好标签
labels.extend([1.0]*len(image_ad))

train_data=[]
train_y=[]
train_data = image[:2400]
train_y=labels[:2400]
train_data.extend(image[2669:5100])
train_y.extend(labels[2669:5100])
train_data=np.array(train_data)
train_data= train_data.reshape(train_data.shape[0],4,2048)
print(train_data.shape)

test_data=[]
test_y=[]
test_data = image[2200:2669]
test_y=labels[2200:2669]
test_data.extend(image[4987:5549])
test_y.extend(labels[4987:5549])
test_data=np.array(test_data)
test_data = test_data.reshape(test_data.shape[0],4,2048)
print(test_data.shape)
train_label=train_y
test_label=test_y

def get_activations(model, inputs, print_shape_only=False, layer_name=None):
    # Documentation is available online on Github at the address below.
    # From: https://github.com/philipperemy/keras-visualize-activations
    print('----- activations -----')
    activations = []
    inp = model.input
    if layer_name is None:
        outputs = [layer.output for layer in model.layers]
    else:
        outputs = [layer.output for layer in model.layers if layer.name == layer_name]  # all layer outputs
    funcs = [K.function([inp] + [K.learning_phase()], [out]) for out in outputs]  # evaluation functions
    layer_outputs = [func([inputs, 1.])[0] for func in funcs]
    for layer_activations in layer_outputs:
        activations.append(layer_activations)
        if print_shape_only:
            print(layer_activations.shape)
        else:
            print(layer_activations)
    return activations
INPUT_DIM = 2048
TIME_STEPS = 4
# if True, the attention vector is shared across the input_dimensions where the attention is applied.
APPLY_ATTENTION_BEFORE_LSTM = False
SINGLE_ATTENTION_VECTOR = False


def attention_3d_block(inputs):
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)
    a = Reshape((input_dim, TIME_STEPS))(a) # this line is not useful. It's just to know which dimension is what.
    a = Dense(TIME_STEPS, activation='softmax')(a)
    if SINGLE_ATTENTION_VECTOR:
        a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
        a = RepeatVector(input_dim)(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    output_attention_mul = Multiply(name="attention_mul")([inputs, a_probs])
    return output_attention_mul

def model_attention_applied_after_lstm():
    inputs = Input(shape=(TIME_STEPS, INPUT_DIM,))
    print(inputs)
    lstm_units = 32
    lstm_out1 = LSTM(lstm_units, return_sequences=True)(inputs)
    lstm_out2 = LSTM(16, return_sequences=True)(lstm_out1)
    lstm_out=LSTM(lstm_units, return_sequences=True)(lstm_out2)
    attention_mul = attention_3d_block(lstm_out)
    attention_mul = Flatten()(attention_mul)
    output = Dense(1, activation='sigmoid')(attention_mul)
    model = Model(input=[inputs], output=output)
    return model

def model_attention_applied_before_lstm():
    inputs = Input(shape=(TIME_STEPS, INPUT_DIM,))
    attention_mul = attention_3d_block(inputs)
    lstm_units = 32
    attention_mul = LSTM(lstm_units, return_sequences=False)(attention_mul)
    output = Dense(1, activation='sigmoid')(attention_mul)
    model = Model(input=[inputs], output=output)
    return model

if __name__ == '__main__':
    inputs_1=np.array(train_data)
    outputs=np.array(train_label)

    if APPLY_ATTENTION_BEFORE_LSTM:
        m = model_attention_applied_before_lstm()
    else:
        m = model_attention_applied_after_lstm()

    # m.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    m.compile(optimizer = optimizers.RMSprop(lr=0.0001, rho=0.9, epsilon=1e-06), loss='binary_crossentropy', metrics=['accuracy'])
    print(m.summary())

    m.fit([inputs_1], outputs, epochs=50, batch_size=16)

    score = m.evaluate(np.array(test_data), np.array(test_label), batch_size=16)
    print(score)


#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/5/20 4:13 PM
# @Author  : yangsen
# @Site    : 
# @File    : kexue.py
# @Software: PyCharm
import json
from tqdm import tqdm
import os, re
import numpy as np
import pandas as pd

mode = 0
min_count = 2
char_size = 128
maxlen = 256

# 读取数据，排除“其他”类型
D = pd.read_csv('../ccks2019_event_entity_extract/event_type_entity_extract_train.csv', encoding='utf-8', header=None)
D = D[D[2] != u'其他']
# D = D[D[1].str.len() <= maxlen]


# 类别映射到数值
if not os.path.exists('classes.json'):
    id2class = dict(enumerate(D[2].unique()))
    class2id = {j:i for i,j in id2class.items()}
    json.dump([id2class, class2id], open('classes.json', 'w'))
else:
    id2class, class2id = json.load(open('classes.json'))


# 排除异常数据，如果 内容中找不到 公司名则不作为训练集
# 排除了4条记录。
train_data = []
for t,c,n in zip(D[1], D[2], D[3]):
    start = t.find(n)
    if start != -1:
        train_data.append((t, c, n))


# 字符转id
if not os.path.exists('all_chars_me.json'):
    chars = {}
    for d in tqdm(iter(train_data)):
        for c in d[0]:
            chars[c] = chars.get(c, 0) + 1
    chars = {i:j for i,j in chars.items() if j >= min_count}
    id2char = {i+2:j for i,j in enumerate(chars)} # 0: mask, 1: padding
    char2id = {j:i for i,j in id2char.items()}
    json.dump([id2char, char2id], open('all_chars_me.json', 'w'))
else:
    id2char, char2id = json.load(open('all_chars_me.json'))


# 训练集乱序序列
np.random.seed(0)
random_order = list(range(len(train_data)))
np.random.shuffle(random_order)


# 训练与测试集分离
dev_data = [train_data[j] for i, j in enumerate(random_order) if i % 5 == mode]
train_data = [train_data[j] for i, j in enumerate(random_order) if i % 5 != mode]

# 评估集
D = pd.read_csv('../ccks2019_event_entity_extract/event_type_entity_extract_eval.csv', encoding='utf-8', header=None)
test_data = []
for id,t,c in zip(D[0], D[1], D[2]):
    test_data.append((id, t, c))

from keras.preprocessing.sequence import pad_sequences


def gene_train(data):
    idxs = range(len(data))

    X, C, S1, S2 = [], [], [], []
    for i in idxs:
        d = data[i]
        text = d[0]
        x = [char2id.get(c, 1) for c in text]
        c = class2id[d[1]]
        s1, s2 = np.zeros(len(text)), np.zeros(len(text))
        start = text.find(d[2])
        end = start + len(d[2]) - 1
        s1[start] = 1
        s2[end] = 1
        X.append(x)
        C.append(c)
        S1.append(s1)
        S2.append(s2)
    X = pad_sequences(X, maxlen=maxlen)
    C = np.array(C)
    S1 = pad_sequences(S1, maxlen=maxlen)
    S2 = pad_sequences(S2, maxlen=maxlen)
    return X, C, S1, S2

X, C, S1, S2 = gene_train(train_data)

test_X, test_C, test_S1, test_S2 = gene_train(dev_data)


from keras.layers import *
from keras.models import Model
import keras.backend as K
from keras.callbacks import Callback
from keras.optimizers import Adam


# model

x_in = Input(shape=(None,)) # 待识别句子输入
c_in = Input(shape=(1,)) # 事件类型
s1_out = Input(shape=(None,)) # 实体左边界（标签）
s2_out = Input(shape=(None,)) # 实体右边界（标签）

x, c = x_in, c_in

# 在第二维度插入，对数值进行截断
# x_mask = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'))(x)

# embedding 字符
x = Embedding(len(id2char)+2, char_size//2)(x)
# embedding 类别
c = Embedding(len(class2id), char_size//2)(c)

# 很奇怪等于啥也没做。
# c = Lambda(lambda x: x[0] * 0 + x[1])([x, c])

# embedding 相加
x = Add()([x, c])
x = Dropout(0.2)(x)

# 与x_mask相乘。
# x = Lambda(lambda x: x[0] * x[1])([x, x_mask])

# lstm
x = Bidirectional(LSTM(char_size//2, return_sequences=False))(x)

# 左边界
x1 = Dense(char_size, use_bias=False, activation='relu')(x)
o1 = Dense(maxlen, use_bias=False, activation='softmax')(x1)

# 右边界
x2 = Dense(char_size, use_bias=False, activation='relu')(x)
o2 = Dense(maxlen, use_bias=False, activation='softmax')(x2)


train_model = Model(inputs=[x_in, c_in], outputs=[o1, o2])
# loss1 = K.mean(K.categorical_crossentropy(s1_out, o1, from_logits=True))
# loss2 = K.mean(K.categorical_crossentropy(s2_out, o2, from_logits=True))
# loss = loss1 + loss2

# train_model.add_loss(loss)
train_model.compile(optimizer=Adam(), loss='categorical_crossentropy')
train_model.summary()

history = train_model.fit([X, C], [S1, S2],
                    epochs=10,
                    batch_size=64,
                    verbose=1)


test_X, test_C, test_S1, test_S2 = gene_train(dev_data)

pred_S1, pred_S2 = train_model.predict([test_X, test_C])
ps1 = pred_S1.argmax(axis=1)
ps2 = pred_S2.argmax(axis=1)

ts1 = test_S1.argmax(axis=1)
ts2 = test_S2.argmax(axis=1)

for i in range(10):
    text = dev_data[i][0]
    y = text[ts1[i]:ts2[i]]
    p = text[ps1[i]:ps2[i]]
    print(f"{y}\t{dev_data[i][2]}\t{text}")


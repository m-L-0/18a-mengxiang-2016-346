# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 18:07:16 2018

@author: 天响之城
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier



#将数据进行处理，添加标签，合并9类别数据
data1 = np.array(pd.read_csv('train/01.csv'))
data2 = np.array(pd.read_csv('train/02.csv'))
data3 = np.array(pd.read_csv('train/03.csv'))
data4 = np.array(pd.read_csv('train/04.csv'))
data5 = np.array(pd.read_csv('train/05.csv'))
data6 = np.array(pd.read_csv('train/06.csv'))
data7 = np.array(pd.read_csv('train/07.csv'))
data8 = np.array(pd.read_csv('train/08.csv'))
data9 = np.array(pd.read_csv('train/09.csv'))
data = np.vstack((data1,data2,data3,data4,data5,data6,data7,data8,data9))

#标准化数据并存储
#data1 = data[:,:-1]
from sklearn import preprocessing
data1 = preprocessing.StandardScaler().fit_transform(data[:,:-1])
#data = preprocessing.MinMaxScaler().fit_transform(new_datawithlabel_array[:,:-1])
label = data[:,-1]
#label = np_utils.to_categorical(label, 9)

X = np.array(data1)
Y = np.array(list(label))
encoder = LabelEncoder()
encoded_Y = encoder.fit_transform(Y)
# convert integers to dummy variables (one hot encoding)
dummy_y = np_utils.to_categorical(encoded_Y)

X_train, X_test, y_train, y_test = train_test_split(X, dummy_y, test_size=0.25, random_state=0)

from keras.callbacks import EarlyStopping
earlystop = EarlyStopping(monitor='val_acc', patience=5, verbose=0, mode='auto')

def baseline_model():
    model = Sequential()
    model.add(Dense(output_dim=128, input_dim=200, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output_dim=256, input_dim=128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output_dim=9, input_dim=256, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

seed = 15
np.random.seed(seed)

#from keras.wrappers.scikit_learn import KerasRegressor
#estimator = KerasRegressor(build_fn=baseline_model, epochs=50, batch_size=32)
estimator = KerasClassifier(build_fn=baseline_model, epochs=100, batch_size=32)

estimator.fit(X_train, y_train,callbacks=[earlystop])

pred = estimator.predict(X_test)

# inverse numeric variables to initial categorical labels
init_lables = encoder.inverse_transform(pred)

# k-fold cross-validate

kfold = KFold(n_splits=8, shuffle=True, random_state=seed)
results = cross_val_score(estimator, X, dummy_y, cv=kfold)
print(np.average(results))

'''
r_test = pd.read_csv('C:/Users/hasee/Desktop/three/01.csv',skiprows = 1,header=None)
data10 = preprocessing.StandardScaler().fit_transform(r_test)

pred1 = estimator.predict(data10)
init_lables = encoder.inverse_transform(pred1)

rep1 = [2 if x==0 else x for x in pred1]
rep2 = [3 if x==1 else x for x in rep1]
rep3 = [5 if x==2 else x for x in rep2]
rep4 = [6 if x==3 else x for x in rep3]
rep5 = [8 if x==4 else x for x in rep4]
rep6 = [10 if x==5 else x for x in rep5]
rep7 = [11 if x==6 else x for x in rep6]
rep8 = [12 if x==7 else x for x in rep7]
rep9 = [14 if x==8 else x for x in rep8]


import csv
with open('C:/Users/hasee/Desktop/three/02.csv',"w") as csvfile: 
    writer = csv.writer(csvfile)
    #先写入columns_name
    #writer.writerow(rep9)
    writer.writerow(pred1)
    csvfile.close()
'''
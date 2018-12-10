# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 17:11:18 2018

@author: 天响之城
"""

'''读取.mat文件转化为csv格式'''

import scipy.io as sio
import pandas as pd

dataFile = 'C:/Users/hasee/Desktop/three//data_test_final.mat'
data = sio.loadmat(dataFile)
print(data)

data1 = data['data_test_final']

dfdata = pd.DataFrame(data1)

datapath1 ='C:/Users/hasee/Desktop/three/01.csv'

dfdata.to_csv(datapath1,index = False)
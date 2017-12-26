# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 16:35:49 2017

@author: zhang
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 20:02:51 2017

@author: zhang
"""


import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.svm import SVR 
from sklearn.externals import joblib
from SVMtrain import load
import os

    
def predict(model,logdata, array, plot_pre,filename):
    narray = np.array(array) 
    logdata[:,7] = np.log10(logdata[:,7])#把电阻率曲线取对数
    test_data = logdata[:,narray]#选取特征，组成最后的输入数据    
    model = joblib.load(model)#加载已有模型
    test_y = model.predict(test_data)
    test_y[np.where(test_y<0)]=0
    total = len(test_y) 
    predict_data = np.column_stack((logdata[:,0],np.array(test_y).reshape((total, 1))))
    
    np.savetxt(filename, predict_data,fmt = "%.4f", delimiter = "\t")
    
    #存曲线图
    
    if plot_pre  == 1:
        fig_pre = plt.figure(6)
        plt.plot(predict_data[:,0], predict_data[:,1])
        plt.xlabel("depth(m)")
        plt.ylabel("toc(%)")
        
        picname =  filename+".png"
        fig_pre.savefig(picname, dpi=600)
        plt.close(1)



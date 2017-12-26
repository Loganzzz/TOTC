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
from sklearn import linear_model
import os
import time
import json


def load(filename):
    data = np.loadtxt(filename,skiprows = 1)
    return data

def generate_point(logdata, tocdata):
    (m, n) = np.shape(tocdata)
    (p, q) = np.shape(logdata)
    logres = np.zeros((m, q))  #m*q
    for i in range(m):
        log_depth = logdata[:, 0]
        idx = np.where(np.abs(log_depth-tocdata[i, 0])<=0.5)  #改变测井求平均值的深度的范围, 重要!!
        depth_mat = np.mat(logdata[idx, :])   #取得深度附近的点
        logres[i, :] = np.mean(depth_mat, axis = 0)
    logres[:, 0] = tocdata[:, 0]
    res = np.mat(np.column_stack((logres, tocdata[:, 1])))#拼接
    return res

def shuffle_data(data):
    (m, n) = np.shape(data)
    test_m = int(m/5)
    cv_m = int(m/5)
    np.random.shuffle(data)
    test_data = data[:test_m,:]
    cv_data = data[test_m-1:(test_m+cv_m), :]    #可能越界
    train_data = data[(test_m+cv_m-1):-1, :]
    return train_data, cv_data, test_data
    
def mdoel_choose(train_data, cv_data, model_map):
    plt.scatter(range(np.shape(cv_data)[0]), cv_data[:,-1],s = 10)
    C = np.array([0.00001,0.00003,0.0001,0.0003,0.001,0.003,0.01,0.03,0.1,0.3,1,3,10,30,100,300,1000])
    gamma = np.array([0.00001,0.00003,0.0001,0.0003,0.001,0.003,0.01,0.03,0.1,0.3,1,3,10,30,100,300,1000])
    cost_J_min = 100000
    model = None
    c_optim = 0
    gamma_optim = 0
    plt.figure(1)
    for i in C:              #通过交叉验证集找合适的C和gamma
        for j in gamma:
            svr_rbf = SVR(kernel = "linear", C = i, gamma = j)
            rbf_model = svr_rbf.fit(train_data[:, 0:-1], train_data[:,-1])
            cv_pre = np.transpose(np.asmatrix(rbf_model.predict(cv_data[:, 0:-1]))) #把array转换成列向量         
            cv_toc = cv_data[:,-1]
  
            plt.plot(range(np.shape(cv_data)[0]), cv_pre)
            cost_J = np.transpose((cv_pre - cv_toc)) * (cv_pre - cv_toc)

            if cost_J<cost_J_min:
                cost_J_min = cost_J
                model = rbf_model
                c_optim = i
                gamma_optim = j
    if not os.path.exists("model"):
        os.mkdir("model")
    systime = time.strftime("%Y%m%d%H%M%S", time.localtime())
    
    #保存模型的名称
    modelpath = "model/"
    modelname = "model_"+systime+".pkl"
    joblib.dump(model,modelpath + modelname)
    plt.close(1)
    #保存模型所使用的测井曲线
    if os.path.exists("mmap.json"):
        f = open("mmap.json","r")  #读写模式打开
        mmap = json.load(f)
        f.close()
        f = open("mmap.json","w")
        mmap[modelname] = model_map
        json.dump(mmap,f)
        f.close()
    else:
        mmap = {}
        mmap[modelname] =  model_map
        f = open("mmap.json","w")
        json.dump(mmap,f)
        f.close()
    return model, c_optim, gamma_optim

def process(logdata,tocdata,tup,plot_pr,plot_es,plot_line, loglist): 
    logdata[:,7] = np.log10(logdata[:,7])   #把电阻率曲线取对数
    logdata = moving_avr(logdata, 3)
    data = generate_point(logdata, tocdata) #选择合适的数据点
    
    tup = np.array(tup)
    input_data = np.column_stack((data[:,tup], data[:,-1]))  #选取特征，组成最后的输入数据
    #input_data = np.column_stack((data[:,tup], np.log2(data[:,-1])))  #toc取log，选取特征，组成最后的输入数据
    
    (train_data, cv_data, test_data) = shuffle_data(input_data)
    (model, C, gamma) = mdoel_choose(train_data, cv_data,loglist)  #训练模型
    test_y = model.predict(test_data[:, 0:-1])
    error_1 = np.where(np.abs(np.transpose(np.asmatrix(test_y))-test_data[:,-1])<1)
    error_2 = np.where(np.abs(np.transpose(np.asmatrix(test_y))-test_data[:,-1])<2)
    error_3 = np.where(np.abs(np.transpose(np.asmatrix(test_y))-test_data[:,-1])<3)
    error_4 = np.where(np.abs(np.transpose(np.asmatrix(test_y))-test_data[:,-1])<0.5)
    print("C的最优值为",C, "   gamma最优值为", gamma)
    print("测试集中误差小于 3 的个数为：%d"% np.shape(error_3)[1])
    print("测试集中误差小于 2 的个数为：%d"% np.shape(error_2)[1])
    print("测试集中误差小于 1 的个数为：%d"% np.shape(error_1)[1])
    print("测试集中误差小于 0.5 的个数为：%d"% np.shape(error_4)[1])
    print("总测试个数为：%d"% np.shape(test_data)[0])
    
    if not os.path.exists("error_anylisis"):
        os.mkdir("error_anylisis")
    
    if plot_line == 1:     
        fig_line = plt.figure(2)
        plt.scatter(range(np.shape(test_data)[0]), test_data[:,-1],s = 10)
        plt.plot(range(np.shape(test_data)[0]), test_y)
        fig_line.savefig("error_anylisis/line.png", dpi=600)
        plt.close(2)
        
    if plot_pr == 1:
        linearclf = linear_model.LinearRegression()
        linearclf.fit(test_data[:,-1], test_y)
        linear_y = linearclf.predict(test_data[:,-1])
        fig_pr = plt.figure(4)
        plt.scatter(np.asarray(test_data[:,-1]),test_y)
        plt.plot(np.asarray(test_data[:,-1]), linear_y, C = "red")
        plt.xlabel('real toc/%')
        plt.ylabel('predict toc/%')
        fig_pr.savefig("error_anylisis/pr.png", dpi=600)
        plt.close(4)
    
    if plot_es == 1:
        error_val = np.linspace(0, 4, 40)
        test_total = np.shape(test_data)[0]
        rate = np.zeros((len(error_val), 1))
        
        for i in range(len(error_val)):
            right_num = np.shape(np.where(np.abs(np.transpose(np.asmatrix(test_y))-test_data[:,-1])<error_val[i]))[1]
            rate[i] = right_num/test_total
        
        fig_es = plt.figure(3)
        plt.plot(error_val, rate)
        plt.title("error visualizing")
        plt.xlabel(r"error_log")
        plt.ylabel(r"stimulate probility/%")
        fig_es.savefig("error_anylisis/es.png", dpi=600)
        plt.close(3)

def moving_avr(X, step):
    '''输入的X为np的多维数组，分别对每一列进行移动平均'''
    shape= np.shape(X)
    res = X.copy()
    M = shape[0]   #行数
    begin = int(step/2)
    end = M - begin
    
    if len(shape) == 1 or shape[1] == 1:
        for i in range(begin, end):
            res[i] = np.mean(X[i-begin: i+begin+1], axis = 0)
    else:
        for i in range(begin, end):
            res[i, :] = np.mean(X[i-begin: i+begin+1, :], axis = 0)
    return res

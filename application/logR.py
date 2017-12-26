# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 08:55:43 2017

@author: zhang
"""

import numpy as np

def load(filename):
    data = np.loadtxt(filename,skiprows = 1)
    return data

def logR(filename, logdata, base_ac='', base_rt='', k=0.02, lom=15, base_toc=0):
    '''
    δlogR法计算
    '''
    logdata = load(logdata)
    lom = float(lom)
    base_toc = float(base_toc)
    #电阻率取对数
    logdata[:,7] = np.log10(logdata[:,7])

    #计算logR
    rt = logdata[:,7]
    ac = logdata[:,1]
    #归一化,确定基线
    ac_norm = ac/600
    rt_norm = (rt+1)/4
    base_index = np.where(np.abs(1-(ac_norm+rt_norm))<0.01)
    if (base_ac==''):
        base_ac = np.mean(ac[base_index])
    else:
        base_ac = float(base_ac)
    if (base_rt==''):
        base_rt = np.mean(rt[base_index])
    else:
        base_rt = np.log10(float(base_rt))
        
    print("预测参数,AC基线:%s，RT基线：%s..."% (base_ac, 10**base_rt))

    logR = rt - base_rt + k * (ac - base_ac)
    #计算toc
    toc_pred = logR * (10 ** (2.297- 0.1688*lom)) + base_toc
    
    toc_pred[np.where(toc_pred<0)]=0
    total = len(toc_pred) 
    predict_data = np.column_stack((logdata[:,0],np.array(toc_pred).reshape((total, 1))))
    np.savetxt(filename, predict_data,fmt = "%.4f", delimiter = "\t")
    return base_ac,10**base_rt
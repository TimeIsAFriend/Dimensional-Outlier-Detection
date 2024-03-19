#!/usr/bin/env python
# coding: utf-8

# In[22]:


import numpy as np
from sklearn.ensemble import IsolationForest
from copy import copy, deepcopy


def KOBES(X,k):
    #output: outlier score
    dim_outlier_score=np.zeros((X.shape[0],X.shape[1]))
    for i in range(X.shape[1]):
        x=X[:,[i]].reshape((1,X.shape[0]))[0]
        idx_x=np.argsort(x)
        sorted_x=x[idx_x]
        sorted_x_app=np.r_[[sorted_x[0]]*k, sorted_x,[sorted_x[-1]]*k]
        s_m,s_p=np.sum(sorted_x_app[:k]),np.sum(sorted_x_app[k+1:2*k+1])
        for j in range(k,x.shape[0]+k):
            dim_outlier_score[idx_x[j-k]][i]=(s_p-s_m)/k
            if j <x.shape[0]+k-1:
                s_m,s_p=s_m-sorted_x_app[j-k]+sorted_x_app[j],s_p-sorted_x_app[j+1]+sorted_x_app[j+k+1]
    return np.mean(dim_outlier_score,axis=1)

def KOBE(X,k):
    #output: outlier score
    dim_outlier_score=np.zeros((X.shape[0],X.shape[1]))
    for i in range(X.shape[1]):
        x=X[:,[i]].reshape((1,X.shape[0]))[0]
        idx_x=np.argsort(x)
        sorted_x=x[idx_x]
        sorted_x_app=np.r_[[sorted_x[0]]*k, sorted_x,[sorted_x[-1]]*k]
        for j in range(k,x.shape[0]+k):
            s_m,s_p=sorted_x_app[j-k],sorted_x_app[j+k]
            dim_outlier_score[idx_x[j-k]][i]=(s_p-s_m)#/k
    return np.mean(dim_outlier_score,axis=1)

def KOBESPlus(X,k):
    #output: outlier score
    den=np.zeros((X.shape[0],X.shape[1]))
    dim_outlier_score=np.zeros((X.shape[0],X.shape[1]))
    for i in range(X.shape[1]):
        x=X[:,[i]].reshape((1,X.shape[0]))[0]
        idx_x=np.argsort(x)
        sorted_x=x[idx_x]
        sorted_x_app=np.r_[[sorted_x[0]]*k, sorted_x,[sorted_x[-1]]*k]
        s_m,s_p=np.sum(sorted_x_app[:k]),np.sum(sorted_x_app[k+1:2*k+1])
        for j in range(k,x.shape[0]+k):
            den[idx_x[j-k]][i]=(s_p-s_m)/(2*k)
            if j <x.shape[0]+k-1:
                s_m,s_p=s_m-sorted_x_app[j-k]+sorted_x_app[j],s_p-sorted_x_app[j+1]+sorted_x_app[j+k+1]
        tem_den=den[:,[i]].reshape((1,X.shape[0]))[0]
        tem_den=np.r_[[tem_den[0]]*k, tem_den,[tem_den[-1]]*k]
        s=np.sum(tem_den[0:2*k+1])
        for j in range(k,x.shape[0]+k):
            dim_outlier_score[j-k][i]=s/(2*k+1)
            if j<x.shape[0]+k-1:
                s=s-tem_den[j-k]+tem_den[j+k+1]
    return np.mean(dim_outlier_score,axis=1)
    
def KOBEPlus(X,k):
    #output: outlier score
    outlier_score=np.zeros(X.shape[0])
    for i in range(X.shape[1]):
        x=X[:,[i]].reshape((1,X.shape[0]))[0]
        idx_x=np.argsort(x)
        sorted_x=x[idx_x]
        sorted_x_app=np.r_[[sorted_x[0]]*k, sorted_x,[sorted_x[-1]]*k]
        t=np.zeros(k+X.shape[0])
        s=0
        for j in range(k,x.shape[0]+k):
            s_m,s_p=sorted_x_app[j-k],sorted_x_app[j+k]
            t[j]=(s_p-s_m)#/k
            if j>=2*k:
                if j==2*k:
                    s+=np.sum(t)
                else:
                    s+=t[j]-t[j-k-3]
                outlier_score[idx_x[j-2*k]]+=s
            else:
                t[j-k]=t[k]
        for j in range(x.shape[0]+k,x.shape[0]+2*k):
                s+=t[-1]-t[j-k-3]
                outlier_score[idx_x[j-2*k]]+=s
    return outlier_score/X.shape[1]

def DDM(X,iteration):
    #output: outlier score
    if iteration>0:
        def MDist(X):
            m=np.median(X,axis=0)
            return abs(X-m)
        for i in range(iteration):
            X=MDist(X)
    return np.mean(X,axis=1)


def DI(X,t):
    #output: outlier score
    def MDist(X):
        return abs(X-np.median(X,axis=0))
    if t>0:
        for i in range(t):
            X=MDist(X)
    return -IsolationForest().fit(X).score_samples(X)


def KI(X,k,t):
    #output: outlier score
    den2=np.zeros(X.shape)
    for i in range(X.shape[1]):
        x=X[:,[i]].reshape((1,X.shape[0]))[0]
        idx_x=np.argsort(x)
        idx_x_0=deepcopy(idx_x)
        for iteration in range(t):
            sorted_x=x[idx_x]
            sorted_x_app=np.r_[[sorted_x[0]]*k, sorted_x,[sorted_x[-1]]*k]
            for j in range(k,x.shape[0]+k):
                den2[idx_x[j-k]][i]= sorted_x_app[j+k]-sorted_x_app[j-k]
            if t>1:
                x=den2[:,[i]].reshape((1,X.shape[0]))[0]
                idx_x=np.argsort(x)
    return -IsolationForest().fit(X).score_samples(X)



# In[ ]:





# In[ ]:





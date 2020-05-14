# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 17:02:17 2019

@author: smcmlab
"""

import matplotlib.pyplot as plt
import numpy as np

pop=20
pc=0.75
pm=0.0075
def obj(x1,x2):
    return x1*np.sin(x1)+x2*np.sin(5*x2)
def encode(x):
    b=[]
    for j in range(len(x)):
        s=8
        a=[]
        for i in range(10):
            a.append(int(x[j]/s))
            if (int(x[j]/s)>0):
                x[j]=x[j]%s
            s/=2
        b.append(np.array(a))
    return b
def fit(x):
    x1,x2=x
    return obj(x1,x2)    
def decode(x):
    b=[]
    for j in range(len(x)):
        s=8
        a=0
        for i in range(10):
            a+=x[j,i]*s
            s/=2
        b.append(a)
    return np.array(b)
    
def init():
    x1=np.random.random_sample([pop,1])*10
    x2=np.random.random_sample([pop,1])*2+4
    return x1,x2
def mut(x):
    size=len(x)*10
    tar=np.random.randint(size, size=int(size*pm))
    for i in range(len(tar)):
        loc1=int(tar[i]/10)
        loc2=tar[i]%10
        x[loc1,loc2]=np.abs(x[loc1,loc2]-1)
    return x


def sel(x,lab):
    sel=x.argsort(axis=0)[int(pc*len(x)):]
    return encode(lab[0][sel]),encode(lab[1][sel])
def corss(x):
    tar=len(x)
    while len(x)<pop:
        temp=int(np.random.randint(7)+2)
        temp1=int(np.random.randint(tar))
        temp2=int(np.random.randint(tar))
        x.append(np.array([*x[temp1][:temp], *x[temp2][temp:]] ))
        x.append(np.array([*x[temp2][:temp], *x[temp1][temp:]] ))
    return x[:pop]
def check(x):
    x1,x2=x
    x3=x1[np.logical_and(np.logical_and(x2>4,x2<6),np.logical_and(x1>0,x1<10))]
    x4=x2[np.logical_and(np.logical_and(x2>4,x2<6),np.logical_and(x1>0,x1<10))]
    return x3,x4
log_1=[]
log_2=[]
log_3=[]

data=init()
fitness=fit(data)
log_1.append(fitness.max())
log_2.append(fitness.min())
log_3.append(fitness.mean())

for fk in range(50):
    sur=sel(fitness,data)
    corss_1=corss(sur[0])
    corss_2=corss(sur[1])
    data=decode(mut(np.array(corss_1))),decode(mut(np.array(corss_1)))
    data=check(data)
    fitness=fit(data)
    log_1.append(fitness.max())
    log_2.append(fitness.min())
    log_3.append(fitness.mean())


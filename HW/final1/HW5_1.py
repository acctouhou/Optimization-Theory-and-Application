# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 02:38:08 2020

@author: Acc
"""

from scipy.optimize import leastsq
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares
from sko.GA import GA

def func(x):
    x1=x[0]
    x2=x[1]
    return (21.5+x1*np.sin(4*np.pi*x1)+x2*np.sin(20*np.pi*x2))
from hyperas import optim
from hyperas.distributions import choice, uniform
from hyperopt import Trials, STATUS_OK, tpe

def data():
    t1=np.loadtxt('t1').reshape(-1,2)
    t2=np.loadtxt('t2').reshape(-1,1)
    
    return t2,t1
def create_model(t2,t1):


    h1={{choice([20,30,40,50])}}
    h2={{choice([20,30,40,50])}}
    it={{choice([20,30,40,50])}}
    
    net = prn.CreateNN([1,h1,h2,2])
    
    net=prn.train_LM(t2.T,t1.T,net,verbose=False,k_max=it,E_stop=1e-5)
    pred=prn.NNOut(t2.T,net).T
    loss=np.mean((pred-t1)**2)
    return {'loss': loss, 'status': STATUS_OK, 'model': net}
import pyrenn as prn

ans_tar =np.array([11.8759,5.7745])
ga_com=[]
hy=[]
lead=[]
from tqdm import tqdm 

for gg in tqdm(range(50)):
    ga = GA(func=func, n_dim=2, size_pop=50, max_iter=1, precision=1e-5,lb=[-3.5,4.1], ub=[12.1,5.8])
    b1=np.array([-3.5,4.1])
    b2=np.array([12.1,5.8])
    ga1 = GA(func=func, n_dim=2, size_pop=50, max_iter=1, precision=1e-5,lb=[-3.5,4.1], ub=[12.1,5.8])
    
    fk1=[]
    fk2=[]
    fk3=[]
    fk4=[]
    coll_t1=[]
    coll_t2=[]
    
    
    
    for i in range(60):
        best_x, best_y = ga.run(1)
        best_11, best_12 = ga1.run(1)
        print(best_y)
        print(best_12)
        t1=ga.X
        t2=ga.Y
        if i<11:
            coll_t1.append(t1)
            coll_t2.append(t2)
        else:
            best_model=prn.train_LM(t2.T,t1.T,best_model,verbose=False,k_max=10,E_stop=1e-8)
        
        if i==10:
            t1=np.array(coll_t1).reshape(-1,2).T
            t2=np.array(coll_t2).reshape(-1,1).T
            np.savetxt('t1',t1)
            np.savetxt('t2',t2)
            best_run, best_model = optim.minimize(model=create_model,
                                              data=data,
                                              algo=tpe.suggest,
                                              max_evals=10,
                                              trials=Trials())
        print(i+1)
        if i>9:
            temp=t2.min()-1e-3
            new=(prn.NNOut(np.array([[temp]]),best_model)).T
            nt=func(new[0])
            print(nt)
            fk1.append(nt)
            fk2.append(new)
        
        
        if i%10==0 and i!=0 and i<50:
            half=(b2-b1)/2
            new=new[0]-b1
            tar=new>half
            print(tar)
            if tar[0]:
                b1[0]=half[0]+b1[0]
            else:
                b2[0]=half[0]+b1[0]
            if tar[1]:
                b1[1]=half[1]+b1[1]
            else:
                b2[1]=half[1]+b1[1]
                
            print(b1)
            print(b2)
            ga = GA(func=func, n_dim=2, size_pop=50, max_iter=50, precision=1e-5,lb=b1, ub=b2)
            print('####################################################################')
    
        
        
        
        fk4.append(best_12)
        fk3.append(best_y)
        
        #%%
    
    ga_com.append(fk4)
    hy.append(fk3)
    lead.append((ans_tar>b1)*(ans_tar<b2))
    plt.plot(fk3,label='Hybrid=%.3f'%(min(fk3)))
    plt.plot(fk4,label='GA=%.3f'%(min(fk4)))
    plt.legend(fontsize=18)
    plt.savefig('other_%d.png'%(gg))
    plt.clf()
    

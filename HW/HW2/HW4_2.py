# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 20:45:05 2019

@author: Acc
"""


import matplotlib.pyplot as plt
import numpy as np

from sko.PSO import PSO

def obj(p):
    x,y=p
    return -np.exp(0.2*((x-1)**2+(y-1)**2)**0.5+np.cos(2*x)+np.sin(2*x))


from sko.GA import GA

def en(code):
    pop,it,w,c1,c2=code
    pso  = PSO(func=obj, dim=2,pop=int(pop), max_iter=int(it), lb=[-5,-5], ub=[5,5], w=w, c1=c1, c2=c2)
    pso.run()
    return pso.gbest_y
    
    
ga = GA(func=en, n_dim=5, size_pop=10, max_iter=50, lb=[1,1,0,0,0], ub=[100,100,1,1,1], precision=1)
    
best_x, best_y = ga.run()
#%%
result=np.array(ga.all_history_Y)
np.repeat(np.arange(50),10)
plt.scatter(np.repeat(np.arange(50),10),result.reshape(-1,1),label='raw_data')
plt.plot(result.min(axis=1),'--',c='r',lw=5,label='best( %.5f )'%(best_y))
plt.legend(fontsize=16)
plt.xlabel('iteration',fontsize=16)
plt.ylabel('fitness',fontsize=16)
plt.title('PSO+GA',fontsize=16)
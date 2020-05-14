# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 21:13:06 2019

@author: Acc
"""

import matplotlib.pyplot as plt
import numpy as np
def obj(p):
    x,y=p
    return 0.5+(np.sin(x**2+y**2)-0.5)/(1+0.1*(x**2+y**2))
from sko.SA import SA
from sko.GA import GA

def en(code):
    pop,s1,s2,t_max,t_min,l,stay=code
    sa  = SA(func=obj, dim=2,pop=pop,x0=[s1,s2], T_max=10**t_max, T_min=10**t_min, L=l, max_stay_counter=stay)
    best_x, best_y = sa.run()
    return best_y
    
    
ga = GA(func=en, n_dim=7, size_pop=50, max_iter=10, lb=[1,-3,-3,-7,-12,1,1], ub=[100,3,3,0,-7,500,500], precision=1e-7)
    
best_x, best_y = ga.run()
#%%
result=np.array(ga.all_history_Y)
np.repeat(np.arange(10),50)
plt.scatter(np.repeat(np.arange(10),50),result.reshape(-1,1),label='raw_data')
plt.plot(result.min(axis=1),'--',c='r',lw=5,label='best( %.5f )'%(best_y))
plt.legend(fontsize=16)
plt.xlabel('iteration',fontsize=16)
plt.ylabel('fitness',fontsize=16)
plt.title('SA+GA',fontsize=16)
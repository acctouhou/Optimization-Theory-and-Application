import matplotlib.pyplot as plt
import numpy as np



x=np.loadtxt('x_data')
y=np.loadtxt('y_data').reshape(-1,21,4)
#%%
yy=y[:,:,[1,3]]
from sklearn import preprocessing
def norm(data):
    a= preprocessing.StandardScaler().fit(data)
    d=a.transform(data)
    m=a.mean_
    s=a.scale_
    v=a.var_
    return d,m,v,s
d1,m1,v1,s1=norm(yy.reshape(-1,2))
d2,m2,v2,s2=norm(x)
d1=d1.reshape(-1,42)


#%%
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split

def rx(x):
    return x/s2*v2+m2
def ry(x):
    return (x.reshape(-1,2)/s1*v1+m1).reshape(-1,42)
def data():
    x=np.loadtxt('x_data')
    y=np.loadtxt('y_data').reshape(-1,21,4)
    yy=y[:,:,[1,3]]
    from sklearn import preprocessing
    def norm(data):
        a= preprocessing.StandardScaler().fit(data)
        d=a.transform(data)
        m=a.mean_
        s=a.scale_
        v=a.var_
        return d,m,v,s
    d1,m1,v1,s1=norm(yy.reshape(-1,2))
    d2,m2,v2,s2=norm(x)
    d1=d1.reshape(-1,42)
    x_train, x_test, y_train, y_test = train_test_split(d2, d1, test_size=0.5, random_state=42)
    return x_train, x_test, y_train, y_test
from hyperas import optim
from hyperas.distributions import choice, uniform
from hyperopt import Trials, STATUS_OK, tpe
def create_model(x_train, x_test, y_train, y_test):
    def rx(x):
        return x/s2*v2+m2
    def ry(x):
        return (x.reshape(-1,2)/s1*v1+m1).reshape(-1,42)
    l1={{choice([64,128])}}
    l2={{choice([64,128,256,512])}}
    l3={{choice([64,128])}}
    activation={{choice(['relu','tanh'])}}
    alpha={{choice([1e-3,1e-4,1e-5])}}
    
    nn=MLPRegressor((l1,l2,l3),verbose=False,alpha=alpha,activation=activation)
    nn.fit(x_train[:400],y_train[:400])
    loss=((np.mean((ry(nn.predict(x_test))-ry(y_test))**2))**0.5)
    return {'loss': loss, 'status': STATUS_OK, 'model': nn}
best_run, best_model = optim.minimize(model=create_model,
                                              data=data,
                                              algo=tpe.suggest,
                                              max_evals=20,
                                              trials=Trials())

#%%
predict=ry(best_model.predict(d2[:100])).reshape(-1,21,2)

import scipy.optimize as optimize
from sko.GA import GA



def func(x):
    log=[]
    for i in range(50):
        try:
            f1 = y[i,:,0]
            t1 = predict[i,:,0]
            def func(t, a, b,c,d):
                return a + x[0]*b*(t**1)+x[1]*c*(t**2)+x[2]*(t**3)+x[3]*c*np.exp(t*d)
            
            popt, pcov = optimize.curve_fit(func, f1, t1, maxfev=500,method='lm')
            log.append((np.mean((func(f1, *popt)-t1)**2))**0.5)
        except:
            log.append(1e3)
    for i in range(50):
        try:
            f1 = y[i,:,2]
            t1 = predict[i,:,1]
            def func(t, a, b,c,d):
                return a + x[0]*b*(t**1)+x[1]*c*(t**2)+x[2]*(t**3)+x[3]*c*np.exp(t*d)
            
            popt, pcov = optimize.curve_fit(func, f1, t1, maxfev=500,method='lm')
            log.append((np.mean((func(f1, *popt)-t1)**2))**0.5)
        except:
            log.append(1e3)
    return sum(log)/200

ga = GA(func=func, n_dim=4, size_pop=40, max_iter=0, precision=1,lb=[0,0,0,0], ub=[1,1,1,1])
for it in range(8):
    print(it+1)
    best_x, best_y = ga.run(1)
#%%
i=5
f1 = y[i,:,0]
t1 = predict[i,:,0]
def func(t, a, b,c,d):
    return a + best_x[0]*b*(t**1)+best_x[1]*c*(t**2)+best_x[2]*(t**3)+best_x[3]*c*np.exp(t*d)
            
popt, pcov = optimize.curve_fit(func, f1, t1, maxfev=500,method='lm')

gg=(np.mean((func(f1, *popt)-t1)**2))**0.5

t = np.linspace(0, 4,1000)
plt.plot( t,func(t, *popt), label="Fitted Curve\nRMSE=%e"%(gg))
plt.scatter(y[i,:,0],y[i,:,1],c='red',label='ANS')
plt.scatter(y[i,:,0],predict[i,:,0],c='g',label='NN result')
plt.legend(loc='upper left')
#%%
i=5
f1 = y[i,:,2]
t1 = predict[i,:,1]

def func(t, a, b,c,d):
    return a + best_x[0]*b*(t**1)+best_x[1]*c*(t**2)+best_x[2]*(t**3)+best_x[3]*c*np.exp(t*d)
            
popt, pcov = optimize.curve_fit(func, f1, t1, maxfev=500,method='lm')

gg=(np.mean((func(f1, *popt)-t1)**2))**0.5

t = np.linspace(0, 4,1000)
plt.plot( t,func(t, *popt), label="Fitted Curve\nRMSE=%e"%(gg))
plt.scatter(y[i,:,2],y[i,:,3],c='red',label='ANS')
plt.scatter(y[i,:,2],predict[i,:,1],c='g',label='NN result')

plt.legend(loc='upper left')

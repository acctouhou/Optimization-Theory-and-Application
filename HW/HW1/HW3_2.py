import numpy as np
mu, sigma = 0, 0.01
orx=np.loadtxt('denoise.m')


s1=np.random.normal(mu, orx.std()*0.3, 1000)
s2=np.random.normal(mu, orx.std()*0.1, 1000)

x=orx+s1+s2
#%%
import matplotlib.pyplot as plt

d1=-np.eye(999,1000)+np.concatenate((np.zeros([999,1]),np.eye(999,999)),axis=1)
d2=np.eye(998,1000)-2*np.concatenate((np.zeros([998,1]),np.eye(998,998),np.zeros([998,1])),axis=1)+np.concatenate((np.zeros([998,2]),np.eye(998,998)),axis=1)
d3=-np.eye(997,1000)+3*np.concatenate((np.zeros([997,1]),np.eye(997,997),np.zeros([997,2])),axis=1)-3*np.concatenate((np.zeros([997,2]),np.eye(997,997),np.zeros([997,1])),axis=1)+np.concatenate((np.zeros([997,3]),np.eye(997,997)),axis=1)



lan=10**2
x_cor1=np.dot(x,np.linalg.inv(np.eye(1000,1000)+lan*np.dot(d1.T,d1)))
lan=10**6
x_cor2=np.dot(x,np.linalg.inv(np.eye(1000,1000)+lan*np.dot(d2.T,d2)))
lan=10**7
x_cor3=np.dot(x,np.linalg.inv(np.eye(1000,1000)+lan*np.dot(d3.T,d3)))


plt.clf()
plt.plot(s1,alpha=0.5,label='noise 1')
plt.plot(s2,alpha=0.5,label='noise 2')
plt.plot(orx,label='raw data',c='black')
plt.legend(fontsize=16)

plt.plot(x,alpha=0.1,c='b')

x_cor=(x_cor1+x_cor2+x_cor3)/3
plt.plot((x_cor1+x_cor2+x_cor3)/3,'r')
plt.plot(x,alpha=0.5,label='raw signal')
plt.plot(x_cor-x,'r',alpha=0.5,label='noise')
plt.title('Combine',fontsize=16)
plt.plot(x_cor,'black',label='reconstructed')
plt.legend(fontsize=16)

import numpy as np
mu, sigma = 0, 0.01

x=np.loadtxt('denoise.m')
import matplotlib.pyplot as plt

d=-np.eye(999,1000)+np.concatenate((np.zeros([999,1]),np.eye(999,999)),axis=1)

tar=np.float32(np.arange(-2,6))
for i in range(len(tar)):
        
    lan=10**tar[i]
    x_cor=np.dot(x,np.linalg.inv(np.eye(1000,1000)+lan*np.dot(d.T,d)))
    
    plt.plot(x,alpha=0.5,label='raw signal')
    plt.plot(x_cor-x,'r',label='noise')
    plt.title('lambda=10^%d'%(tar[i]),fontsize=16)
    plt.plot(x_cor,'black',label='reconstructed')
    plt.legend(fontsize=16)
    plt.savefig('%d.png'%(tar[i]))
    plt.clf()

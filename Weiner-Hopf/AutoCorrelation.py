import numpy as np 
from GaussianWhiteNoise import sin,noise
import matplotlib.pyplot as plt

def autoCorrelation(x):
    N = len(x)
    rx_k = np.zeros(N)
    for k in range (0,N):
        correlation = 0
        for n in range (k,N):
            correlation = correlation + x[n] * np.conj(x[n-k])
        rx_k[k] = (correlation/(N-abs(k)))
    return rx_k

def autoCorrelation_np(x):
    N = len(x)
    r = np.correlate(x, x, mode='full')  # full correlation
    return r[N-1:] / np.arange(N, 0, -1)  # normalize like sum / (N-k)


#Visualization of Correlated and Uncorrelated Inputs
correlated_sample = sin
uncorrelated_sample = noise
r_k = autoCorrelation(correlated_sample)
k = np.arange(0,len(r_k))
fig, axs = plt.subplots(1,2,figsize = (8,8))
axs[0].plot(k,r_k)
axs[0].set_xlabel("k")
axs[0].set_ylabel("Correlation")
axs[0].set_title("Correlated Sample")
r_k = autoCorrelation(uncorrelated_sample)
k = np.arange(0,len(r_k))
axs[1].plot(k,r_k)
axs[1].set_xlabel("k")
axs[1].set_ylabel("Correlation")
axs[1].set_title("Uncorrelated Sample")
plt.tight_layout()
plt.show()
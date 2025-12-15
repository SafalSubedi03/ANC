import numpy as np 
from SignalGeneration import sin,noise,xn
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


auto_Correlation = autoCorrelation(xn)
#Visualization of Correlated and Uncorrelated Inputs
if __name__ == "__main__":
    correlated_sample = sin[:-100]
    uncorrelated_sample = noise[:-100]
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
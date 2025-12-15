import numpy as np 
import matplotlib.pyplot as plt 
from SignalGeneration import noise,xn

def crossCorelation(x_n,d_n):
    if len(x_n) != len(d_n): 
        raise ValueError("The input and the desired output length mush match")
    N = len(x_n)
    rdx = np.zeros(N)
    for k in range (0,N):
        sum = 0
        for n in range (k,N):
            sum =  sum + d_n[n] * np.conj(x_n[n-k])
        rdx[k] = sum/(N - abs(k))
    return rdx 

desiredSignal = 0.8*xn + 0.6*np.roll(xn, 5) +0.9*np.roll(xn, 29) + 0.1*np.random.randn(len(xn))
desiredSignalwithNoNoise = 0.8*xn + 0.6*np.roll(xn, 5) +0.9*np.roll(xn, 29)
Cross_Corelation = crossCorelation(xn,desiredSignal)

# Example usage
if __name__ == "__main__":
    x_n = noise
    d_n = 0.8*x_n + 0.6*np.roll(x_n, 5) +0.9*np.roll(x_n, 29) + 0.1*np.random.randn(len(x_n))  # desired output

    r_dx = crossCorelation(x_n, d_n)
    lags = np.arange(len(r_dx))

    plt.figure(figsize=(8,4))
    plt.plot(lags, r_dx)
    plt.title("Cross-correlation r_dx[k]")
    plt.xlabel("Lag k")
    plt.ylabel("r_dx[k]")
    plt.grid(True)
    plt.show()

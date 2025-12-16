import numpy as np 
import matplotlib.pyplot as plt
from SignalGeneration import xn
from Cross_Correlation import desiredSignal,desiredSignalwithNoNoise

def updateWeight(wk, mu, x, e, n):
    for k in range(len(wk)):
        if n - k >= 0:
            wk[k] = wk[k] + mu * x[n - k] * np.conj(e)
    return wk

    

def compute_yn(wn,M,xn,n):
    sum = 0
    for k in range (0,M):
        sum = sum + wn[k] * xn[n-k]
    return sum 

N = 3000
xn = xn[:N]
desiredSignal = desiredSignal[:N]
# desiredSignalwithNoNoise = desiredSignalwithNoNoise[:N]
#Parameters
M = 50 
u = 0.02


wk = [0 for k in range(M)]
yn = []
en = []

for n in range(0,N):
    yn.append(compute_yn(wk,M,xn,n))
    en.append(desiredSignal[n] - yn[-1])
    wk = updateWeight(wk,u,xn,en[n],n)





fig, axs = plt.subplots(2, 2, figsize=(6,6))
time_axis = np.arange(0,len(xn))
# Time domain
axs[0,0].plot(time_axis, xn)
axs[0,0].set_xlabel("Time [s]")
axs[0,0].set_ylabel("Amplitude")
axs[0,0].set_title("Time Domain Input Signal")

axs[0,1].plot(time_axis, yn)
axs[0,1].set_xlabel("Time [s]")
axs[0,1].set_ylabel("Amplitude")
axs[0,1].set_title("Time Domain Output Signal")


axs[1,0].plot(time_axis, en)
axs[1,0].set_xlabel("Time [s]")
axs[1,0].set_ylabel("Amplitude")
axs[1,0].set_title("Time Domain error Signal")


axs[1,1].plot(time_axis, desiredSignal)
axs[1,1].set_xlabel("Time [s]")
axs[1,1].set_ylabel("Amplitude")
axs[1,1].set_title("Time Domain desired Signal")






plt.tight_layout()
plt.show()
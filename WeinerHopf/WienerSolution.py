import numpy as np
import matplotlib.pyplot as plt

from Cross_Correlation import Cross_Corelation, desiredSignal
from AutoCorrelation import auto_Correlation
from SignalGeneration import xn

def build_R(r_x, M):
    """Construct the M×M Toeplitz autocorrelation matrix."""
    R = np.zeros((M, M))
    for i in range(M):
        for j in range(M):
            R[i, j] = r_x[abs(i - j)]
    return R

def build_p(r_dx, M):
    """Construct the M×1 cross-correlation vector."""
    return r_dx[:M].astype(float)

def bestestimate_dn(w, x, M):
    """Compute the filter output d̂[n] using M-tap FIR filter."""
    N = len(x)
    d_hat = np.zeros(N)
    for n in range(M - 1, N):
        for k in range(M):
            d_hat[n] += w[k] * x[n - k]
    return d_hat

M = 40

R = build_R(auto_Correlation, M)
p = build_p(Cross_Corelation, M)

try:
    w_opt = np.linalg.solve(R, p)
except np.linalg.LinAlgError:
    w_opt = np.linalg.pinv(R).dot(p)

d_hat = bestestimate_dn(w_opt, xn, M)
error = desiredSignal - d_hat

fig, axs = plt.subplots(2, 2, figsize=(6, 6))

k = np.arange(len(xn))
axs[0, 0].plot(k, xn)
axs[0, 0].set_title("Input Signal x[n]")

k = np.arange(len(desiredSignal))
axs[0, 1].plot(k, desiredSignal)
axs[0, 1].set_title("Desired Signal d[n]")

axs[1, 0].plot(k, d_hat)
axs[1, 0].set_title("Estimated Signal d̂[n]")

axs[1, 1].plot(k, error)
axs[1, 1].set_title("Error Signal e[n]")

plt.tight_layout()
plt.show()

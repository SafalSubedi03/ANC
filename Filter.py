import numpy as np
import matplotlib.pyplot as plt

Fs = 1500
Ts = 1/Fs
N = Fs
n = np.arange(N)
t = n * Ts

x_n = 3*np.sin(2*np.pi*25*t) + 1.5*np.sin(2*np.pi*700*t)

class LowPass:
    def LowPassImpulseResponse(self, wc, M):
        if M % 2 == 0:
            raise ValueError("Filter length M must be odd")
        h_n = np.zeros(M)
        k = (M - 1) // 2
        for n in range(M):
            l = n - k
            if l == 0:
                h_n[n] = wc / np.pi
            else:
                h_n[n] = np.sin(wc * l) / (np.pi * l)
        return h_n

    def Calculate_yn(self, xn, wc, M):
        h_n = self.LowPassImpulseResponse(wc, M)
        return np.convolve(xn, h_n, mode="same")

lpf = LowPass()
fc = 100
wc = 2*np.pi*fc / Fs
M = 201

y_n = lpf.Calculate_yn(x_n, wc, M)

# ----------------------------
# FFT of signals
# ----------------------------
X_w = np.fft.fft(x_n)
Y_w = np.fft.fft(y_n)
freq_axis = np.arange(len(X_w)) * Fs / len(X_w)

# ----------------------------
# FFT of filter impulse response
# ----------------------------
h_n = lpf.LowPassImpulseResponse(wc, M)
H_w = np.fft.fft(h_n, N)  # zero-pad to N points
freq_axis_filter = np.arange(N) * Fs / N

# ----------------------------
# Plots
# ----------------------------
fig, axes = plt.subplots(3,2, figsize=(14,12))

# Time domain signals
axes[0,0].plot(t, x_n)
axes[0,0].set_xlabel("Time [s]")
axes[0,0].set_ylabel("Amplitude")
axes[0,0].set_title("Original Signal")

axes[0,1].plot(t, y_n)
axes[0,1].set_xlabel("Time [s]")
axes[0,1].set_ylabel("Amplitude")
axes[0,1].set_title("Filtered Signal")

# Frequency domain signals
N_half = N // 2
axes[1,0].plot(freq_axis[:N_half], np.abs(X_w[:N_half]))
axes[1,0].set_xlabel("Frequency [Hz]")
axes[1,0].set_ylabel("|X(f)|")
axes[1,0].set_title("FFT of Original Signal")

axes[1,1].plot(freq_axis[:N_half], np.abs(Y_w[:N_half]))
axes[1,1].set_xlabel("Frequency [Hz]")
axes[1,1].set_ylabel("|Y(f)|")
axes[1,1].set_title("FFT of Filtered Signal")

# Filter Bode plots: magnitude and phase
axes[2,0].plot(freq_axis_filter[:N_half], 20*np.log10(np.abs(H_w[:N_half])))
axes[2,0].set_xlabel("Frequency [Hz]")
axes[2,0].set_ylabel("Magnitude [dB]")
axes[2,0].set_title("Filter Magnitude Response")

axes[2,1].plot(freq_axis_filter[:N_half], np.angle(H_w[:N_half]))
axes[2,1].set_xlabel("Frequency [Hz]")
axes[2,1].set_ylabel("Phase [radians]")
axes[2,1].set_title("Filter Phase Response")

plt.tight_layout()
plt.show()

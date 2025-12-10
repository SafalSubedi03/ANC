import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import remez, freqz

# ----------------------------
# Signal setup
# ----------------------------
Fs = 1500
Ts = 1/Fs
N = Fs
n = np.arange(N)
t = n * Ts

# Test signal: combination of low and high frequencies
x_n = 3*np.sin(2*np.pi*25*t) + 1.5*np.sin(2*np.pi*700*t)

# ----------------------------
# Window-based FIR class
# ----------------------------
class LowPassWindow:
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
        y_n = np.convolve(xn, h_n, mode="same")
        return y_n, h_n

# ----------------------------
# Equiripple FIR class
# ----------------------------
class LowPassEquiripple:
    def impulseResponse(self, fc, M, Fs, delta_f=0.05, weight_pass=1, weight_stop=10):
        # Convert cutoff and transition to normalized frequency (0-0.5)
        f_c = fc / Fs
        bands = [0, f_c, min(f_c + delta_f, 0.5), 0.5]
        desired = [1, 0]
        weight = [weight_pass, weight_stop]
        # fs argument is sampling frequency
        h = remez(M, bands, desired, weight=weight, fs=1.0)  # bands already normalized
        return h

    def Calculate_yn(self, xn, h):
        return np.convolve(xn, h, mode="same")

# ----------------------------
# Filter parameters
# ----------------------------
fc = 200                    # cutoff frequency (Hz)
wc = 2*np.pi*fc / Fs        # for window-based FIR
M = 201                     # filter length

# ----------------------------
# Window FIR filtering
# ----------------------------
lpf_window = LowPassWindow()
y_window, h_window = lpf_window.Calculate_yn(x_n, wc, M)

# ----------------------------
# Equiripple FIR filtering
# ----------------------------
lpf_equi = LowPassEquiripple()
h_equi = lpf_equi.impulseResponse(fc, M, Fs, delta_f=0.05, weight_pass=1, weight_stop=10)
y_equi = lpf_equi.Calculate_yn(x_n, h_equi)

# ----------------------------
# Frequency response using freqz
# ----------------------------
w_window, H_window = freqz(h_window, worN=1024, fs=Fs)
w_equi, H_equi = freqz(h_equi, worN=1024, fs=Fs)

# ----------------------------
# FFT of signals for comparison
# ----------------------------
def compute_fft(signal, N, Fs):
    X_w = np.fft.fft(signal, N)
    freq_axis = np.arange(N) * Fs / N
    return freq_axis, X_w

freq_axis, X_w = compute_fft(x_n, N, Fs)
_, Y_window_w = compute_fft(y_window, N, Fs)
_, Y_equi_w = compute_fft(y_equi, N, Fs)
N_half = N // 2

# ----------------------------
# Plots
# ----------------------------
fig, axes = plt.subplots(4, 2, figsize=(14, 14))

# Time domain signals
axes[0,0].plot(t, x_n)
axes[0,0].set_title("Original Signal")
axes[0,0].set_xlabel("Time [s]")
axes[0,0].set_ylabel("Amplitude")

axes[0,1].plot(t, y_window)
axes[0,1].set_title("Window FIR Filtered Signal")
axes[0,1].set_xlabel("Time [s]")
axes[0,1].set_ylabel("Amplitude")

axes[1,0].plot(t, y_equi)
axes[1,0].set_title("Equiripple FIR Filtered Signal")
axes[1,0].set_xlabel("Time [s]")
axes[1,0].set_ylabel("Amplitude")

# Frequency domain signals (FFT)
axes[1,1].plot(freq_axis[:N_half], np.abs(X_w[:N_half]))
axes[1,1].set_title("FFT of Original Signal")
axes[1,1].set_xlabel("Frequency [Hz]")
axes[1,1].set_ylabel("|X(f)|")

axes[2,0].plot(freq_axis[:N_half], np.abs(Y_window_w[:N_half]))
axes[2,0].set_title("FFT of Window FIR Filtered Signal")
axes[2,0].set_xlabel("Frequency [Hz]")
axes[2,0].set_ylabel("|Y(f)|")

axes[2,1].plot(freq_axis[:N_half], np.abs(Y_equi_w[:N_half]))
axes[2,1].set_title("FFT of Equiripple FIR Filtered Signal")
axes[2,1].set_xlabel("Frequency [Hz]")
axes[2,1].set_ylabel("|Y(f)|")

# Filter magnitude responses
axes[3,0].plot(w_window, 20*np.log10(np.abs(H_window)))
axes[3,0].set_title("Window FIR Magnitude Response (dB)")
axes[3,0].set_xlabel("Frequency [Hz]")
axes[3,0].set_ylabel("Magnitude [dB]")
axes[3,0].grid(True)

axes[3,1].plot(w_equi, 20*np.log10(np.abs(H_equi)))
axes[3,1].set_title("Equiripple FIR Magnitude Response (dB)")
axes[3,1].set_xlabel("Frequency [Hz]")
axes[3,1].set_ylabel("Magnitude [dB]")
axes[3,1].grid(True)

plt.tight_layout()
plt.show()

# ----------------------------
# Overlay magnitude response comparison
# ----------------------------
plt.figure(figsize=(10,5))
plt.plot(w_window, 20*np.log10(np.abs(H_window)), label="Window FIR")
plt.plot(w_equi, 20*np.log10(np.abs(H_equi)), label="Equiripple FIR")
plt.title("FIR Filter Magnitude Response Comparison")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Magnitude [dB]")
plt.grid(True)
plt.legend()
plt.show()

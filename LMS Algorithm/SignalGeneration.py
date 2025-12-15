import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

# --- Generate Gaussian White Noise ---
def generate_gaussian_white_noise(mean=0, std_dev=1, num_samples=1000):
    if num_samples <= 0:
        raise ValueError("Number of samples must be positive.")
    if std_dev < 0:
        raise ValueError("Standard deviation must be non-negative.")
    return np.random.normal(loc=mean, scale=std_dev, size=num_samples)


def sinusoids(x,w = 2*np.pi*20):
    return np.sin(w*x)

# --- Parameters ---
Fs = 1000      # Sampling frequency
N = 500         # Number of samples (1 second)
endpoint = 4   # seconds
time_axis = np.arange(0, endpoint,1/Fs)
noise = generate_gaussian_white_noise(0, 1, len(time_axis))
sin = sinusoids(time_axis)
xn = sin + 0.3 * noise


if __name__ == "__main__":

    # --- Manual Periodogram (one-sided) ---
    ft_xn = np.fft.fft(xn)
    freqs = np.fft.fftfreq(N, 1/Fs)
    Pxx = (1/Fs) * np.abs(ft_xn)**2
    Pxx_one = Pxx[:N//2].copy()
    Pxx_one[1:-1] *= 2             # double positive frequencies except DC & Nyquist
    freqs_one = freqs[:N//2]

    # --- Welch PSD ---
    f_welch, Pxx_welch = welch(xn, fs=Fs, window='hann', nperseg=256, noverlap=128, scaling='density')

    # --- Plotting ---
    fig, axs = plt.subplots(1, 3, figsize=(15,4))

    # Time domain
    axs[0].plot(time_axis, xn)
    axs[0].set_xlabel("Time [s]")
    axs[0].set_ylabel("Amplitude")
    axs[0].set_title("Time Domain Noise")

    # Manual Periodogram
    axs[1].plot(freqs_one, Pxx_one)
    axs[1].set_xlabel("Frequency [Hz]")
    axs[1].set_ylabel("PSD [V²/Hz]")
    axs[1].set_title("Manual One-sided PSD")

    # Welch
    axs[2].plot(f_welch, Pxx_welch)
    axs[2].set_xlabel("Frequency [Hz]")
    axs[2].set_ylabel("PSD [V²/Hz]")
    axs[2].set_title("Welch PSD (Averaged)")

    plt.tight_layout()
    plt.show()

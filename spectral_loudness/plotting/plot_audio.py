import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy import signal
import numpy as np


class PlotAudio(object):
    def __init__(self, signal=[], fs=48000.0, bit_depth=16):
        self.signal = signal
        self.length = len(self.signal)
        self.fs = fs
        self.bit_depth = bit_depth

    @staticmethod
    def show():
        plt.show()

    def normalize(self):
        return self.signal / (2. ** (self.bit_depth-1))

    def plot_fft(self, title='FFT'):
        sig_norm = self.normalize()

        p = fft(sig_norm)  # take the fourier transform
        n_unique_pts = int(np.ceil((self.length + 1) / 2.0))
        p = p[0:n_unique_pts]
        p = abs(p)

        p = p / float(self.length) # scale by the number of points so that
        # the magnitude does not depend on the length
        # of the signal or on its sampling frequency
        p = p ** 2  # square it to get the power

        # multiply by two (see technical document for details)
        # odd nfft excludes Nyquist point
        if self.length % 2 > 0:  # we've got odd number of points fft
            p[1:len(p)] = p[1:len(p)] * 2
        else:
            p[1:len(p) - 1] = p[1:len(p) - 1] * 2  # we've got even number of points fft

        freq_array = np.arange(0, n_unique_pts, 1.0) * (self.fs / self.length)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.plot(freq_array, 10 * np.log10(p), color='k')
        plt.xlabel('Frequency (kHz)')
        plt.ylabel('Power (dB)')
        ax.axis('tight')
        plt.title(title)
        plt.grid()

    def plot_waveform(self, title='Waveform'):
        sig_norm = self.normalize()
        data_shape = self.signal.shape[0]
        time_array = np.arange(0, data_shape, 1)
        time_array_scaled = time_array / self.fs * self.length
        plt.figure(figsize=(4, 3), dpi=128)
        plt.plot(time_array_scaled, sig_norm, color='r')
        plt.ylabel('Amplitude')
        plt.xlabel('Time (ms)')
        plt.title(title)
        plt.grid()

    def plot_fir_filter(self, coeffs, title='FIR'):
        w, h = signal.freqz(coeffs)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.plot(w/max(w) * self.fs/2, 20 * np.log10(abs(h)), 'b')
        ax.set_xlabel('Frequency [radians / second]')
        ax.set_ylabel('Amplitude [dB]')
        plt.title(title)
        ax.axis('tight')
        ax.grid(which='both', axis='both')

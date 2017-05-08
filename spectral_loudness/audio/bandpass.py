from scipy import signal
import numpy as np


class BandpassFilterBank(object):
    def __init__(self, fs=48000, n_filter=31, numtaps=480):
        self.n_filter = n_filter
        self.numtaps = numtaps
        self.fs = fs
        self.nyq = fs/2
        self.corner_freq = np.geomspace(20, 2e4, self.n_filter)

    def bandpass_filter_coeffs(self, freq_low, freq_high):
        low_norm = freq_low / self.nyq
        high_norm = freq_high / self.nyq
        coeffs = signal.firwin(self.numtaps, [low_norm, high_norm], window='blackmanharris', pass_zero=False)
        return coeffs

    def bandpass_filter(self, freq_low, freq_high, buffer):
        coeffs = self.bandpass_filter_coeffs(freq_low, freq_high)
        output = signal.lfilter(coeffs, 1.0, buffer)
        return output

    def filter_bank(self, buffer):
        filtered_array = np.zeros((self.n_filter, buffer.size))
        for i, freq in enumerate(self.corner_freq):
            if freq == self.corner_freq[-1]:
                break
            filtered_array[i] = self.bandpass_filter(self.corner_freq[i], self.corner_freq[i+1], buffer)
        return filtered_array

from scipy import signal


class BandpassFilterBank(object):
    def __init__(self, fs=48000, n_filter=31, numtaps=1000):
        self.n_filter = n_filter
        self.numtaps = numtaps
        self.fs = fs
        self.nyq = fs/2

    def bandpass_filter_coeffs(self, freq_low, freq_high):
        low_norm = freq_low / self.nyq
        high_norm = freq_high / self.nyq
        coeffs = signal.firwin(self.numtaps, [low_norm, high_norm], window='blackmanharris', pass_zero=False)
        return coeffs

    def filter(self, freq_low, freq_high, input):
        coeffs = self.bandpass_filter_coeffs(freq_low, freq_high)
        output = signal.lfilter(coeffs, 1.0, input)
        return output


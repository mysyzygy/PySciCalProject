# python 3.4.2

import matplotlib.pyplot as plt
from scipy.io import wavfile as wav
from scipy.fftpack import fft
import numpy as np
import os

INPUT_DIR = os.path.join(os.path.dirname(__file__), 'test_files')


def plot_waveform(input_file, window):
    """
    plot_waveform, plots the waveform of a 16 bit wav file 

    Based on http://samcarcagno.altervista.org/blog/basic-sound-processing-python/

    :param input_file: input 
    :param window: number of samples from the start to be evaluated
    :return: None
    """
    fs, data = wav.read(input_file)
    data_ch1 = data[:, 0]
    data_ch1_sliced = data_ch1[:window]
    data_shape = data_ch1_sliced.shape[0]
    data_norm = data_ch1_sliced / (2.**15)
    time_array = np.arange(0, data_shape, 1)
    time_array_scaled = time_array / fs * 1000
    plt.figure(figsize=(4, 3), dpi=128)
    plt.plot(time_array_scaled, data_norm, color='r')
    plt.ylabel('Amplitude')
    plt.xlabel('Time (ms)')
    plt.grid()
    plt.show()


def plot_fft(input_file, window):
    """
    plot_fft, plots the fft of a 16 bit wav file 
    
    Based on http://samcarcagno.altervista.org/blog/basic-sound-processing-python/
    
    :param input_file: input 
    :param window: window size to be evaluated
    :return: None
    """
    fs, data = wav.read(input_file)
    data_ch1 = data[:, 0]
    data_ch1_sliced = data_ch1[:window]
    data_norm = data_ch1_sliced / (2. ** 15)
    n = len(data_norm)
    p = fft(data_norm) # take the fourier transform

    nUniquePts = int(np.ceil((n+1)/2.0))
    p = p[0:nUniquePts]
    p = abs(p)

    p = p / float(n) # scale by the number of points so that
                     # the magnitude does not depend on the length
                     # of the signal or on its sampling frequency
    p = p**2  # square it to get the power

    # multiply by two (see technical document for details)
    # odd nfft excludes Nyquist point
    if n % 2 > 0: # we've got odd number of points fft
        p[1:len(p)] = p[1:len(p)] * 2
    else:
        p[1:len(p) -1] = p[1:len(p) - 1] * 2 # we've got even number of points fft

    freqArray = np.arange(0, nUniquePts, 1.0) * (fs / n)
    plt.plot(freqArray/1000, 10*np.log10(p), color='k')
    plt.xlabel('Frequency (kHz)')
    plt.ylabel('Power (dB)')
    plt.show()


def main():
    input_file = os.path.join(INPUT_DIR, 'test_file1.wav')
    # plot_waveform(input_file, 4800)
    plot_fft(input_file, 48000)

if __name__ == "__main__":
    main()
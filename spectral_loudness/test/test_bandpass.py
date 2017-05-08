import os

import numpy as np
import pytest
from ..audio.bandpass import BandpassFilterBank
from ..plotting.plot_audio import PlotAudio
from scipy.io import wavfile as wav

PLOT = False

INPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'test_files')


def test_bandpass_filter_coeffs():
    bpfb = BandpassFilterBank(numtaps=1000)
    coeffs = bpfb.bandpass_filter_coeffs(100., 200.)
    assert coeffs.size == 1000


def test_fir_filter():
    t = np.linspace(0, 64, 512, endpoint=False)
    sig = np.sin(2 * np.pi * 5. * t)
    bpfb = BandpassFilterBank()
    output = bpfb.bandpass_filter(100.0, 200.0, sig)
    assert max(output) == 0.0003473088433417048
    assert min(output) == -0.0024289363918210041


def test_filter_audio():
    input_file = os.path.join(INPUT_DIR, 'test_file1.wav')
    fs, data = wav.read(input_file)
    data_ch1 = data[:, 0]
    data_sliced = data_ch1[:48000]
    bpfb = BandpassFilterBank()
    output = bpfb.bandpass_filter(400.0, 500.0, data_sliced)

    assert max(output) == 3265.2004911313984
    assert min(output) == -3265.220946076156

    bpfb2 = BandpassFilterBank()
    output2 = bpfb2.bandpass_filter(315.0, 400.0, data_sliced)

    assert max(output2) == 2581.3983869978547
    assert min(output2) == -2555.9894235947404

    if PLOT is True:
        plt = PlotAudio(output, float(fs))
        plt.plot_waveform('Output Waveform')
        plt.plot_fft('Output FFT')

        plt2 = PlotAudio(output2, float(fs))
        plt2.plot_waveform('Output 2 Waveform')
        plt2.plot_fft('Output 2 FFT')
        plt.show()


@pytest.mark.skipif(PLOT is False, reason='plotting is not set to True')
def test_plot_filter():
    bpfb = BandpassFilterBank(numtaps=10000)
    coeffs = bpfb.bandpass_filter_coeffs(8000., 16000.)
    plot = PlotAudio()
    plot.plot_fir_filter(coeffs)
    plot.show()


@pytest.mark.parametrize('n_filter', [16, 31])
@pytest.mark.parametrize('numtaps', [48, 480])
@pytest.mark.parametrize('window', [4800])
def test_filter_bank(n_filter, numtaps, window):
    input_file = os.path.join(INPUT_DIR, 'test_file1.wav')
    fs, data = wav.read(input_file)
    data_ch1 = data[:, 0]
    data_sliced = data_ch1[:4800+(numtaps*2)]
    bpfb = BandpassFilterBank(n_filter=n_filter, numtaps=numtaps)
    freq_values = bpfb.corner_freq
    filtered_array = bpfb.filter_bank(data_sliced)
    assert filtered_array.shape == (n_filter, window + numtaps*2)

    if PLOT is True:
        for i, freq in enumerate(filtered_array):
            freq_slice = freq[numtaps: -numtaps]
            plt = PlotAudio(freq_slice, float(fs))
            plt.plot_waveform('Output Waveform: {0}\n{1}\n{2}'.format(freq_values[i], n_filter, numtaps))
            plt.plot_fft('Output FFT {0}\n{1}\n{2}'.format(freq_values[i], n_filter, numtaps))
        plt.show()


import pytest
import numpy as np
from scipy.io import wavfile as wav
import os
from audio.bandpass import BandpassFilterBank
from plotting.plot_audio import PlotAudio

plot = True

INPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'test_files')


def test_bandpass_filter_coeffs():
    bpfb = BandpassFilterBank()
    coeffs = bpfb.bandpass_filter_coeffs(100., 200.)
    assert coeffs.size == 1000


def test_fir_filter():
    t = np.linspace(0, 64, 512, endpoint=False)
    sig = np.sin(2 * np.pi * 5. * t)
    bpfb = BandpassFilterBank()
    output = bpfb.filter(100.0, 200.0, sig)
    assert max(output) == 0.00066693385673081583
    assert min(output) == -0.0013171638800527343


def test_filter_audio():
    input_file = os.path.join(INPUT_DIR, 'test_file1.wav')
    fs, data = wav.read(input_file)
    data_ch1 = data[:, 0]
    data_sliced = data_ch1[:48000]
    bpfb = BandpassFilterBank()
    output = bpfb.filter(400.0, 500.0, data_sliced)

    assert max(output) == 3235.3047600373138
    assert min(output) == -3235.2912576891777

    bpfb2 = BandpassFilterBank()
    output2 = bpfb2.filter(315.0, 400.0, data_sliced)

    assert max(output2) == 1561.2523501851306
    assert min(output2) == -1547.3280725013785

    if plot is True:
        plt = PlotAudio(output, float(fs))
        plt.plot_waveform('Output Waveform')
        plt.plot_fft('Output FFT')

        plt2 = PlotAudio(output2, float(fs))
        plt2.plot_waveform('Output 2 Waveform')
        plt2.plot_fft('Output 2 FFT')


@pytest.mark.skip
def test_plot_filter():
    bpfb = BandpassFilterBank(numtaps=10000)
    coeffs = bpfb.bandpass_filter_coeffs(8000., 16000.)
    plot = PlotAudio()
    plot.plot_fir_filter(coeffs)


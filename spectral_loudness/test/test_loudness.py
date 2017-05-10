import os
import pytest
import numpy as np
import scipy.signal as signal
from ..plotting import plot_audio
from ..audio.engine import Engine
from ..audio.loudness import Loudness

INPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'test_files')

PLOT = False


@pytest.mark.skipif(PLOT == False, reason='Plotting is set to False')
def test_plot_k_filters():
    a = np.array([1.0, -1.69065929318241, 0.73248077421585])
    b = np.array([1.53512485958697, -2.6916918940638, 1.19839281085285])
    plt = plot_audio.PlotAudio()
    plt.plot_freqz_filter(b, a, 'K-Weight Shelf')
    print('K-Weight Shelf: b:{}\na:{}'.format(b, a))

    a = np.array([1.0, -1.99004745483398, 0.99007225036621])
    b = np.array([1.0, -2.0, 1.0])
    plt = plot_audio.PlotAudio()
    plt.plot_freqz_filter(b, a, 'K-Weight Highpass')
    print('K-Weight Highpass: b:{}\na:{}'.format(b, a))

    plt.show()


def test_k_filters():
    t = np.linspace(0, 64, 512, endpoint=False)
    sig = np.sin(2 * np.pi * 5. * t)
    loudness = Loudness()
    output = loudness.k_weight(sig)
    assert max(output) == 1.0070375121769581
    assert min(output) == -1.0050361882049823


def test_mean_square():
    t = np.linspace(0, 64, 512, endpoint=False)
    sig = np.sin(2 * np.pi * 5. * t)
    loudness = Loudness()
    output = loudness.mean_square(sig)
    assert max(output) == 0.001953125
    assert min(output) == 0.0


def test_mean_square_lufs():
    t = np.linspace(0, 64, 512, endpoint=False)
    sig = np.sin(2 * np.pi * 5. * t)
    loudness = Loudness()
    mean_square = loudness.mean_square(sig)
    output = loudness.lufs(mean_square)
    assert max(output) == -3.4002699609758307


def test_true_peak():
    t = np.linspace(0, 64, 512, endpoint=False)
    sig = np.sin(2 * np.pi * 10. * t)
    loudness = Loudness()
    true_peak = loudness.true_peak(sig)
    assert true_peak == 0.0


@pytest.mark.parametrize('n_filter', [31])
@pytest.mark.parametrize('numtaps', [480])
@pytest.mark.parametrize('buffer_size', [4800])
@pytest.mark.parametrize('wav', ['test_file1.wav'])
def test_loudness(n_filter, numtaps, buffer_size, wav):
    input_file = os.path.join(INPUT_DIR, wav)
    engine = Engine(input_file=input_file, buffer_size=buffer_size,
                    n_filter=n_filter, numtaps=numtaps)
    engine.run()


def test_plot_poly():
    coeffs = np.array([0.0017089843750,
                       -0.0291748046875,
                       -0.0189208984375,
                       -0.0083007812500,
                       0.0109863281250,
                       0.0292968750000,
                       0.0330810546875,
                       0.0148925781250,
                       -0.0196533203125,
                       -0.0517578125000,
                       -0.0582275390625,
                       -0.0266113281250,
                       0.0332031250000,
                       0.0891113281250,
                       0.1015625000000,
                       0.0476074218750,
                       -0.0594482421875,
                       -0.1665039062500,
                       -0.2003173828125,
                       -0.1022949218750,
                       0.1373291015625,
                       0.4650878906250,
                       0.7797851562500,
                       0.9721679687500,
                       0.9721679687500,
                       0.7797851562500,
                       0.4650878906250,
                       0.1373291015625,
                       -0.1022949218750,
                       -0.2003173828125,
                       -0.1665039062500,
                       -0.0594482421875,
                       0.0476074218750,
                       0.1015625000000,
                       0.0891113281250,
                       0.0332031250000,
                       -0.0266113281250,
                       -0.0582275390625,
                       -0.0517578125000,
                       -0.0196533203125,
                       0.0148925781250,
                       0.0330810546875,
                       0.0292968750000,
                       0.0109863281250,
                       -0.0083007812500,
                       -0.0189208984375,
                       -0.0291748046875,
                       0.0017089843750])
    plt = plot_audio.PlotAudio()
    plt.plot_fir_filter(coeffs, title='True Peak Poly')
    plt.show()
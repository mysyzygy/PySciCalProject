import os
import numpy as np
from ..plotting.animate import PlotBarGraph, Histogram, Animate


INPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'test_files')


def test_plot_bar():
    true_peak_array = np.load(os.path.join(INPUT_DIR, 'Calvin_Harris_This_Is_What_You_Came_For_peak.npy'))
    loudness_array = np.load(os.path.join(INPUT_DIR, 'Calvin_Harris_This_Is_What_You_Came_For_loud.npy'))
    corner_freqs = np.geomspace(20, 2e4, loudness_array.shape[1])
    animate = PlotBarGraph(corner_freqs)
    animate.plot_histogram(loudness_array[0], true_peak_array[2])
    animate.show()


def test_simple_animation():
    true_peak_array = np.load(os.path.join(INPUT_DIR, 'Calvin_Harris_This_Is_What_You_Came_For_peak.npy'))
    corner_freqs = np.geomspace(20, 2e4, true_peak_array.shape[1])
    true_peak_hist = Histogram(corner_freqs, facecolor='green', edgecolor='yellow')

    def yield_arrays():
        for buffer in true_peak_array:
            yield buffer

    animate = Animate(true_peak_hist, corner_freqs, yield_arrays())
    animate.run()


def test_run_animation():
    true_peak_array = np.load(os.path.join(INPUT_DIR, 'Calvin_Harris_This_Is_What_You_Came_For_peak.npy'))
    loudness_array = np.load(os.path.join(INPUT_DIR, 'Calvin_Harris_This_Is_What_You_Came_For_loud.npy'))
    corner_freqs = 17
    true_peak_hist = Histogram(corner_freqs, facecolor='r')
    loudness_hist = Histogram(corner_freqs, facecolor='b')
    histograms = [true_peak_hist, loudness_hist]

    def yield_arrays():
        for i, buffer in enumerate(true_peak_array):
            yield buffer, loudness_array[i]

    animate = Animate(histograms, corner_freqs, yield_arrays())
    animate.run()
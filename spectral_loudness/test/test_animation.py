import os
import numpy as np
from ..plotting.animate import Animate


INPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'test_files')


def test_histogram():
    loudness_array = np.load(os.path.join(INPUT_DIR, 'Calvin_Harris_This_Is_What_You_Came_For_loud.npy'))
    true_peak_array = np.load(os.path.join(INPUT_DIR, 'Calvin_Harris_This_Is_What_You_Came_For_peak.npy'))
    corner_freqs = np.geomspace(20, 2e4, loudness_array.shape[1])
    animate = Animate(corner_freqs)
    animate.animate(loudness_array[0], true_peak_array[2])
    animate.show()


def test_run_animation():
    loudness_array = np.load(os.path.join(INPUT_DIR, 'Calvin_Harris_This_Is_What_You_Came_For_loud.npy'))
    true_peak_array = np.load(os.path.join(INPUT_DIR, 'Calvin_Harris_This_Is_What_You_Came_For_peak.npy'))
    corner_freqs = np.geomspace(20, 2e4, loudness_array.shape[1])
    animate = Animate(corner_freqs)
    for i in range(loudness_array.shape[0]):
        animate.animate(loudness_array[i], true_peak_array[i])
        animate.show()
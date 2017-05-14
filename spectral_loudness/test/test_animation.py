import os
import numpy as np
from ..plotting.animate import Animate


INPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'test_files')


def test_histogram():
    loudness_array = np.load(os.path.join(INPUT_DIR, 'Calvin_Harris_This_Is_What_You_Came_For_loud.npy'))
    true_peak_array = np.load(os.path.join(INPUT_DIR, 'Calvin_Harris_This_Is_What_You_Came_For_peak.npy'))
    corner_freqs = np.geomspace(20, 2e4, loudness_array.shape[1])
    animate = Animate(loudness_array[0], true_peak_array[2], corner_freqs)
    animate.show()


def test_animation():
    input_file = os.path.join(INPUT_DIR, 'potc.npy')
    input_array = np.load(input_file)

    animate = Animate(input_array)
    animate.run_animation()

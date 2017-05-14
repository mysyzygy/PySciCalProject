import os
import numpy as np
from ..plotting.animate import Animate


INPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'test_files')


def test_animation():
    input_file = os.path.join(INPUT_DIR, 'potc.npy')
    input_array = np.load(input_file)

    animate = Animate(input_array)
    animate.run_animation()

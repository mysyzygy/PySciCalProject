
"""
==================
Animated histogram
==================
This example shows how to use a path patch to draw a bunch of
rectangles for an animated histogram.
"""

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.path as path
import matplotlib.animation as animation
from matplotlib import gridspec


class Animate:
    def __init__(self, loudness_array, true_peak_array, n_freqs):

        self.loudness_array = loudness_array
        self.true_peak_array = true_peak_array

        ind = np.arange(len(n_freqs))  # the x locations for the groups
        width = 0.67  # the width of the bars

        fig = plt.figure(figsize=(12, 8))
        ax = plt.subplot2grid((1, 1), (0, 0))

        rects1 = ax.bar(ind, 96 + self.true_peak_array, width, bottom=-96, color='r')
        rects2 = ax.bar(ind, 96 + self.loudness_array, width, bottom=-96, color='g')

        # add some text for labels, title and axes ticks
        ax.set_ylabel('dB')
        ax.set_yticks((-96, 0))
        ax.set_title('True Peak and Loudness Measurement')
        ax.set_xticks(ind)
        ax.set_xticklabels(n_freqs.astype(int))
        ax.legend((rects1[0], rects2[0]), ('True Peak', 'Loudness'))

        plt.tight_layout()
        plt.show()

    @staticmethod
    def show():
        plt.show()

    # def animate(self, i):
    #     # simulate new data coming in
    #     #data = np.random.randn(1000)
    #     n, bins = np.histogram(self.data)
    #     top = self.bottom + n
    #     self.verts[1::5, 1] = top
    #     self.verts[2::5, 1] = top
    #     return [self.patch, ]
    #
    # def run_animation(self):
    #     ani = animation.FuncAnimation(self.fig, self.animate, 100, repeat=False, blit=True)
    #     self.show()

import numpy as np
import matplotlib.pyplot as plt
import sys
import matplotlib.patches as patches
import matplotlib.path as path
import matplotlib.animation as animation
from matplotlib import gridspec


class Animate:
    def __init__(self, n_freqs):
        self.n_freqs = n_freqs
        self.ind = np.arange(len(self.n_freqs))  # the x locations for the groups
        self.width = 0.67  # the width of the bars
        self.fig = plt.figure(figsize=(12, 8))
        self.ax = plt.subplot2grid((1, 1), (0, 0))

    def plot_histogram(self, loudness_array, true_peak_array):
        rects1 = self.ax.bar(self.ind, 96 + true_peak_array, self.width, bottom=-96, color='r')
        rects2 = self.ax.bar(self.ind, 96 + loudness_array, self.width, bottom=-96, color='b')
        self.ax.set_ylabel('dB')
        self.ax.set_yticks((-96, 0))
        self.ax.set_title('True Peak and Loudness Measurement')
        self.ax.set_xticks(self.ind)
        self.ax.set_xticklabels(self.n_freqs.astype(int))
        self.ax.legend((rects1[0], rects2[0]), ('True Peak', 'Loudness'))

    @staticmethod
    def show():
        plt.show()
        plt.close()

    def run_animation(self, loudness_array, true_peak_array):
        pass
        # ani = animation.FuncAnimation(self.fig, self.animate, yield_array, blit=True)

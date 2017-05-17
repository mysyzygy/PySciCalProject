import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.path as path
import matplotlib.animation as animation
import time


class PlotBarGraph:
    def __init__(self, corner_freq):
        self.corner_freq = corner_freq.astype(np.int16)
        self.freq_count = np.arange(len(self.corner_freq))  # the x locations for the groups
        self.width = 0.67  # the width of the bars
        self.fig = plt.figure(figsize=(12, 8))
        self.ax = plt.subplot2grid((1, 1), (0, 0))

    def plot_histogram(self, loudness_array, true_peak_array):
        rects1 = self.ax.bar(self.freq_count, 96 + true_peak_array, self.width, bottom=-96, color='r')
        rects2 = self.ax.bar(self.freq_count, 96 + loudness_array, self.width, bottom=-96, color='b')
        self.ax.set_ylabel('dB')
        self.ax.set_yticks((np.linspace(-96, 0, 17)))
        self.ax.set_title('True Peak and Loudness Measurement')
        self.ax.set_xticks(self.freq_count)
        self.ax.set_xticklabels(self.corner_freq)
        self.ax.set_xlabel('Hz')
        self.ax.legend((rects1[0], rects2[0]), ('True Peak', 'Loudness'))

    @staticmethod
    def show():
        plt.show()
        plt.close()


class Histogram:
    def __init__(self, corner_freq, facecolor='green', edgecolor='black'):
        self.data = np.zeros(16)
        self.corner_freq = corner_freq.astype(np.int16)
        self.freq_count = np.arange(self.corner_freq.size + 1)

        # get the corners of the rectangles for the histogram
        self.left = np.array(self.freq_count[:-1])
        self.right = np.array(self.freq_count[1:])
        self.bottom = np.zeros(len(self.left))
        self.top = self.bottom + self.data
        self.nrects = len(self.left)

        self.nverts = self.nrects * (1 + 3 + 1)
        self.verts = np.zeros((self.nverts, 2))
        self.codes = np.ones(self.nverts, int) * path.Path.LINETO
        self.codes[0::5] = path.Path.MOVETO
        self.codes[4::5] = path.Path.CLOSEPOLY
        self.verts[0::5, 0] = self.left
        self.verts[0::5, 1] = self.right
        self.verts[1::5, 0] = self.left
        self.verts[1::5, 1] = self.top
        self.verts[2::5, 0] = self.right
        self.verts[2::5, 1] = self.top
        self.verts[3::5, 0] = self.right
        self.verts[3::5, 1] = self.bottom
        self.barpath = path.Path(self.verts, self.codes)
        self.patch = patches.PathPatch(self.barpath, facecolor=facecolor, edgecolor=edgecolor, alpha=1.0)

START = 0
STOP = 0


def print_time():
    print('loop time: {}'.format(STOP - START))


class Animate:
    def __init__(self, histogram, corner_freq, gen_function,):
        self.histograms = histogram
        self.corner_freq = corner_freq.astype(np.int16)
        self.freq_count = np.arange(len(self.corner_freq))
        self.gen_function = gen_function
        self.fig, self.ax = plt.subplots()

        if isinstance(self.histograms, list):
            for hist in self.histograms:
                self.ax.add_patch(hist.patch)
        else:
            self.ax.add_patch(self.histograms.patch)

        self.ax.set_yticks((np.linspace(0, 96, 17)))
        self.ax.set_title('True Peak and Loudness Measurement')
        self.ax.set_xticks(self.freq_count)
        self.ax.set_xticklabels(self.corner_freq)
        self.ax.set_xlabel('Hz')
        # self.ax.legend((self.histograms[0], self.histograms[1]), ('True Peak', 'Loudness'))

    def update(self, dyn_buffer):
        # simulate new data coming in
        global STOP
        STOP = time.time()
        print_time()
        global START
        START = time.time()


        patches = []
        if isinstance(self.histograms, list):
            for i, hist in enumerate(dyn_buffer):
                dyn_buffer_array = dyn_buffer[i] + 96
                self.histograms[i].verts[1::5, 1] = dyn_buffer_array
                self.histograms[i].verts[2::5, 1] = dyn_buffer_array
                patches.append(self.histograms[i].patch)
        else:
            dyn_buffer_array = dyn_buffer + 96
            self.histograms.verts[1::5, 1] = dyn_buffer_array
            self.histograms.verts[2::5, 1] = dyn_buffer_array
            patches = [self.histograms.patch, ]
        return patches

    def process(self):
            next(self.gen_function)

    def run(self):
        ani = animation.FuncAnimation(self.fig, self.update, self.gen_function, interval=100, blit=True)
        plt.show()
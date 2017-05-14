
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


class Animate:
    def __init__(self, data):

        self.data = data
        self.fig, ax = plt.subplots()

        # histogram our data with numpy
        #data = np.random.randn(1000)
        n, bins = np.histogram(self.data)

        # get the corners of the rectangles for the histogram
        left = np.array(bins[:-1])
        right = np.array(bins[1:])
        self.bottom = np.zeros(len(left))
        top = self.bottom + n
        nrects = len(left)

        # here comes the tricky part -- we have to set up the vertex and path
        # codes arrays using moveto, lineto and closepoly

        # for each rect: 1 for the MOVETO, 3 for the LINETO, 1 for the
        # CLOSEPOLY; the vert for the closepoly is ignored but we still need
        # it to keep the codes aligned with the vertices
        nverts = nrects*(1 + 3 + 1)
        self.verts = np.zeros((nverts, 2))
        codes = np.ones(nverts, int) * path.Path.LINETO
        codes[0::5] = path.Path.MOVETO
        codes[4::5] = path.Path.CLOSEPOLY
        self.verts[0::5, 0] = left
        self.verts[0::5, 1] = self.bottom
        self.verts[1::5, 0] = left
        self.verts[1::5, 1] = top
        self.verts[2::5, 0] = right
        self.verts[2::5, 1] = top
        self.verts[3::5, 0] = right
        self.verts[3::5, 1] = self.bottom

        barpath = path.Path(self.verts, codes)
        self.patch = patches.PathPatch(
            barpath, facecolor='green', edgecolor='yellow', alpha=0.5)
        ax.add_patch(self.patch)

        ax.set_xlim(left[0], right[-1])
        ax.set_ylim(self.bottom.min(), top.max())


    def animate(self, i):
        # simulate new data coming in
        #data = np.random.randn(1000)
        n, bins = np.histogram(self.data)
        top = self.bottom + n
        self.verts[1::5, 1] = top
        self.verts[2::5, 1] = top
        return [self.patch, ]

    def run_animation(self):
        ani = animation.FuncAnimation(self.fig, self.animate, 100, repeat=False, blit=True)
        plt.show()
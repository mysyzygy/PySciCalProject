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
import time

fig, ax = plt.subplots()

# histogram our data with numpy
n = np.zeros(16)
bins = np.arange(0, 17, 1)
# n, bins = np.histogram(data, 16)

# get the corners of the rectangles for the histogram
left = np.array(bins[:-1])
right = np.array(bins[1:])
bottom = np.zeros(len(left))
top = bottom + n
nrects = len(left)

# here comes the tricky part -- we have to set up the vertex and path
# codes arrays using moveto, lineto and closepoly

# for each rect: 1 for the MOVETO, 3 for the LINETO, 1 for the
# CLOSEPOLY; the vert for the closepoly is ignored but we still need
# it to keep the codes aligned with the vertices
nverts = nrects*(1 + 3 + 1)
verts = np.zeros((nverts, 2))
codes = np.ones(nverts, int) * path.Path.LINETO
codes[0::5] = path.Path.MOVETO
codes[4::5] = path.Path.CLOSEPOLY
verts[0::5, 0] = left
verts[0::5, 1] = bottom
verts[1::5, 0] = left
verts[1::5, 1] = top
verts[2::5, 0] = right
verts[2::5, 1] = top
verts[3::5, 0] = right
verts[3::5, 1] = bottom

barpath = path.Path(verts, codes)
patch = patches.PathPatch(
    barpath, facecolor='green', edgecolor='yellow', alpha=0.5)
ax.add_patch(patch)

n = np.zeros(16)
bins = np.arange(0, 17, 1)

# get the corners of the rectangles for the histogram
left = np.array(bins[:-1])
right = np.array(bins[1:])
bottom = np.zeros(len(left))
top = bottom + n
nrects = len(left)

# here comes the tricky part -- we have to set up the vertex and path
# codes arrays using moveto, lineto and closepoly

# for each rect: 1 for the MOVETO, 3 for the LINETO, 1 for the
# CLOSEPOLY; the vert for the closepoly is ignored but we still need
# it to keep the codes aligned with the vertices
nverts1 = nrects*(1 + 3 + 1)
verts1 = np.zeros((nverts1, 2))
codes1 = np.ones(nverts1, int) * path.Path.LINETO
codes1[0::5] = path.Path.MOVETO
codes1[4::5] = path.Path.CLOSEPOLY
verts1[0::5, 0] = left
verts1[0::5, 1] = bottom
verts1[1::5, 0] = left
verts1[1::5, 1] = top
verts1[2::5, 0] = right
verts1[2::5, 1] = top
verts1[3::5, 0] = right
verts1[3::5, 1] = bottom

barpath = path.Path(verts1, codes1)
patch1 = patches.PathPatch(
    barpath, facecolor='blue', edgecolor='yellow', alpha=0.5)
ax.add_patch(patch1)

ax.set_xlim(left[0], right[-1])
ax.set_ylim(0, 100)

START = 0
STOP = 0


def print_time():
    print('loop time: {}'.format(STOP - START))


def animate(i):
    # simulate new data coming in
    global STOP
    STOP = time.time()
    print_time()
    global START
    START = time.time()
    n = np.random.randint(0, 100, 16)
    top = bottom + n
    verts[1::5, 1] = top
    verts[2::5, 1] = top

    n = np.random.randint(0, 100, 16)
    top = bottom + n
    verts1[1::5, 1] = top
    verts1[2::5, 1] = top

    return [patch, patch1]

ani = animation.FuncAnimation(fig, animate, 1000, interval=100, repeat=False, blit=True)
plt.show()
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time

fig, ax = plt.subplots()
xdata, ydata = [np.linspace(0, 20000, 16)], [np.linspace(10, 100, 16)]
ln, = plt.plot([],[], 'r', animated=True)


def init():
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 100)
    return ln,


def update(frame):
    xdata = [np.linspace(0, 20000, 16)]
    ydata = frame
    ln.set_data(xdata, ydata)
    return ln,


def frame_maker():
    while True:
        time.sleep(.8)
        yield np.random.randint(0, 100, 16)



ani = FuncAnimation(fig, update, frames=frame_maker(), init_func=init, blit=True)
plt.show()



# fig, ax = plt.subplots()
# xdata, ydata = [], []
# ln, = plt.plot([], [], 'ro', animated=True)
#
#
# def init():
#     ax.set_xlim(0, 2*np.pi)
#     ax.set_ylim(-1, 1)
#     return ln,
#
#
# def update(frame):
#     xdata.append(frame)
#     ydata.append(np.sin(frame))
#     ln.set_data(xdata, ydata)
#     return ln,
#
# ani = FuncAnimation(fig, update, frames=np.linspace(0, 2*np.pi, 128), init_func=init, blit=True)
# plt.show()
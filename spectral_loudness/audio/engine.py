import numpy as np
from scipy.io import wavfile as wav
import sounddevice as sd
import sys

from .bandpass import BandpassFilterBank
from .loudness import Loudness

import time
try:
    import queue  # Python 3.x
except ImportError:
    import Queue as queue  # Python 2.x
import threading

DEBUG = False


class Engine:
    def __init__(self, input_file, output_file, dyn_rng_type='short', buffer_size=4800,
                 n_filter=16, numtaps=1200, queue_size=20):

        self.input_file = input_file
        self.output_file = output_file
        self.buffer_size = buffer_size
        self.queue_size = queue_size
        self.n_filter = n_filter
        self.numtaps = numtaps
        self.dyn_rng_type = dyn_rng_type

        # generate bandpass and loudness objects
        self.bpfb = BandpassFilterBank(n_filter=self.n_filter, numtaps=self.numtaps)
        self.loudness = Loudness(n_filter=self.n_filter, dyn_rng=self.dyn_rng_type)

        # parse wav file
        self.fs, self.data = wav.read(self.input_file)

        # check that wav file is valid for measurement
        if self.data.shape[1] != 2 or self.fs != 48000 or self.data.dtype != np.int16:
            raise ValueError('spectral loudness only supports stereo 48kHz, 16-bit wav files.')

        # create first buffer
        self.pad = np.zeros((self.numtaps, 2))
        self.data_start = np.append(self.pad, self.data, 0)

        # create final buffer
        self.pad = np.zeros((self.numtaps, 2))
        self.data_end = np.append(self.data, self.pad, 0)

        # create result arrays
        self.dyn_rng_array = np.zeros((round(self.data.shape[0] / self.buffer_size), self.n_filter), dtype=np.float64)
        self.loudness_array = np.zeros((round(self.data.shape[0] / self.buffer_size), self.n_filter), dtype=np.float64)
        self.true_peak_array = np.zeros((round(self.data.shape[0] / self.buffer_size), self.n_filter), dtype=np.float64)

        # setup sounddevice for playback
        self.sd = sd
        self.sd.default.latency = 'high'

        # setup queue for playback
        self.q = queue.Queue(maxsize=self.queue_size)
        self.event = threading.Event()

        # DEBUG start and stop times
        self.stop = 0
        self.start = 0

    def callback(self, outdata, frames, time, status):
        assert frames == self.buffer_size
        if status.output_underflow:
            print('Output underflow: increase blocksize?', file=sys.stderr)
            raise sd.CallbackAbort
        assert not status
        try:
            data = self.q.get_nowait()
        except queue.Empty:
            print('Queue is empty: increase queue size?', file=sys.stderr)
            raise sd.CallbackAbort
        if len(data) < len(outdata):
            outdata[:len(data)] = data
            outdata[len(data):] = b'\x00' * (len(outdata) - len(data))
            raise sd.CallbackStop
        else:
            outdata[:] = data

    def remove_padding(self, buffer):
        return buffer[:, self.numtaps: -self.numtaps]

    def write_file(self):
        if self.output_file.endswith('.npy'):
            np.save(self.output_file, self.dyn_rng_array)
        elif self.output_file.endswith('.txt'):
            np.savetxt(self.output_file, self.dyn_rng_array)
        else:
            raise ValueError('output file format not supported: {}'.format(self.output_file))

    def run(self):
        print('Running spectrum analyzer on input file: {}'.format(self.input_file))

        stream = sd.OutputStream(
            samplerate=self.fs, blocksize=self.buffer_size,
            device=1, channels=2, dtype='int16',
            callback=self.callback, finished_callback=self.event.set)

        # generate number of buffers
        buffer_count = int((self.data.size/2)/self.buffer_size)

        with stream:
            # loop through each buffer
            for buffer in range(buffer_count):
                # DEBUG TIMING TEST
                if DEBUG:
                    self.start = time.time()

                # generate start and stop samples
                buffer_start = self.buffer_size * buffer
                buffer_stop = self.buffer_size * (buffer + 1)

                if buffer is 0:
                    padded_stop = self.numtaps + buffer_stop + self.numtaps

                    # slice buffer with pre and post pad
                    ch1 = self.data_start[: padded_stop, :1]
                    ch2 = self.data_start[: padded_stop, 1:]

                    mono_buffer = (ch1 + ch2)/2
                    # set audio buffer
                    audio_buffer = self.data[buffer_start: buffer_stop]

                elif buffer == buffer_count - 1:
                    padded_start = buffer_start - self.numtaps

                    # slice buffer with pre and post pad
                    ch1 = self.data_end[padded_start:, :1]
                    ch2 = self.data_end[padded_start:, 1:]

                    mono_buffer = (ch1 + ch2) / 2
                    # set audio buffer
                    audio_buffer = self.data[buffer_start: buffer_stop]

                else:
                    # set buffer with pre and post pad
                    padded_start = buffer_start - self.numtaps
                    padded_stop = buffer_stop + self.numtaps

                    # slice buffer with pre and post pad
                    ch1 = self.data[padded_start: padded_stop, :1]
                    ch2 = self.data[padded_start: padded_stop, 1:]

                    mono_buffer = (ch1 + ch2) / 2

                    # set audio buffer
                    audio_buffer = self.data[buffer_start: buffer_stop]

                # add audio buffer to queue
                self.q.put(audio_buffer)

                # create filtered array and remove padding - array size is equal to self.n_filter
                filtered_array_mono = self.remove_padding(self.bpfb.filter_bank(np.ndarray.flatten(mono_buffer)))

                # measure loudness for each filtered array
                result = self.loudness.process(filtered_array_mono)

                # enter dyn_rng value into array
                self.loudness_array[buffer, ] = np.ndarray.flatten(result[0])
                self.true_peak_array[buffer, ] = np.ndarray.flatten(result[1])
                self.dyn_rng_array[buffer,] = np.ndarray.flatten(result[2])

                # DEBUG TIMING TEST
                if DEBUG:
                    self.stop = time.time()
                    print('loop time {}'.format(self.stop - self.start))

                # quit loop on last buffer
                if buffer == buffer_count - 1:
                    break
            self.event.wait()
        self.write_file()

        avg_loudness = np.sum(self.loudness_array, 0) / self.loudness_array.shape[0]
        avg_dyn_rng = np.sum(self.dyn_rng_array, 0) / self.dyn_rng_array.shape[0]
        avg_true_peak = np.sum(self.true_peak_array, 0) / self.true_peak_array.shape[0]

        print('Average Loudness: {}\nAverage Dynamic Range: {}\n Average True Peak: {}'.format(avg_loudness,
                                                                                               avg_dyn_rng,
                                                                                               avg_true_peak))
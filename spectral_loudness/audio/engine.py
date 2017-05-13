import numpy as np
from scipy.io import wavfile as wav
import sounddevice as sd

from .bandpass import BandpassFilterBank
from .loudness import Loudness

import time


class Engine():
    def __init__(self, input_file, buffer_size=4800, n_filter=16, numtaps=1200):
        self.input_file = input_file
        self.buffer_size = buffer_size
        self.n_filter = n_filter
        self.numtaps = numtaps

        self.bpfb = BandpassFilterBank(n_filter=self.n_filter, numtaps=self.numtaps)
        self.fs, self.data = wav.read(self.input_file)
        # check that wav file is stereo
        if self.data.shape[1] != 2 or self.fs != 48000 or self.data.dtype != np.int16:
            raise ValueError('spectral loudness only supports stereo 48kHz, 16-bit wav files.')

        self.pad = np.zeros((self.numtaps, 2))
        self.data_start = np.append(self.pad, self.data, 0)

        self.loudness_ch1 = Loudness(self.n_filter)
        self.loudness_ch2 = Loudness(self.n_filter)

        self.sd = sd
        self.sd.default.channels = 2
        self.sd.default.samplerate = self.fs
        self.sd.default.latency = 'high'

        self.stop = 0
        self.start = 0




    def remove_padding(self, buffer):
        return buffer[:, self.numtaps: -self.numtaps]

    def run(self):
        # print('Running spectrum analyzer on input file: {}'.format(self.input_file))

        # generate number of buffers
        buffer_count = int((self.data.size/2)/self.buffer_size)

        # loop through each buffer
        for buffer in range(buffer_count):
            self.start = time.time()

            # generate start and stop samples
            buffer_start = self.buffer_size * buffer
            buffer_stop = self.buffer_size * (buffer + 1)

            # print(buffer_start, buffer_stop)

            if buffer is 0:

                padded_stop = self.numtaps + buffer_stop + self.numtaps

                # slice buffer with pre and post pad
                ch1 = self.data_start[: padded_stop, :1]
                ch2 = self.data_start[: padded_stop, 1:]

                mono_buffer = (ch1 + ch2)/2

                # set audio buffer
                audio_buffer = self.data[buffer_start: buffer_stop]

            # skip last buffer
            elif buffer == buffer_count - 1:
                break
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

            # play audio back
            # self.sd.play(audio_buffer, blocking=True)

            # create filtered array and remove padding - array size is equal to self.n_filter
            filtered_array_mono = self.remove_padding(self.bpfb.filter_bank(np.ndarray.flatten(mono_buffer)))

            # filtered_array_ch1 = self.remove_padding(self.bpfb.filter_bank(np.ndarray.flatten(ch1)))
            # filtered_array_ch2 = self.remove_padding(self.bpfb.filter_bank(np.ndarray.flatten(ch2)))

            # measure loudness for each filtered array
            mono_result = self.loudness_ch1.process(filtered_array_mono)

            # ch1_result = self.loudness_ch1.process(filtered_array_ch1)
            # ch2_result = self.loudness_ch2.process(filtered_array_ch2)


            # generate mono dynamic range values

            momentary_dyn_rng = (mono_result[4] + mono_result[4]) / 2
            short_term_dyn_rng = (mono_result[5] + mono_result[5]) / 2

            # momentary_dyn_rng = (ch1_result[4] + ch2_result[4]) / 2
            # short_term_dyn_rng = (ch1_result[5] + ch2_result[5]) / 2
            #
            # print('Dynamic Range For Buffer: {}'
            #       '\nMomentary DR: {}'
            #       '\nShort Term DR: {}'.format(buffer, momentary_dyn_rng, short_term_dyn_rng))

            self.stop = time.time()
            print('loop time {}'.format(self.stop - self.start))


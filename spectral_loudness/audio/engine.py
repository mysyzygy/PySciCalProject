import numpy as np
from scipy.io import wavfile as wav

from .bandpass import BandpassFilterBank
from .loudness import Loudness


class Engine():
    def __init__(self, input_file, buffer_size=4800, n_filter=31, numtaps=480):
        self.input_file = input_file
        self.buffer_size = buffer_size
        self.n_filter = n_filter
        self.numtaps = numtaps

    def remove_padding(self, buffer):
        return buffer[:, self.numtaps: -self.numtaps]

    def run(self):
        print('Running analyzer on input file: {}'.format(self.input_file))

        fs, data = wav.read(self.input_file)

        if fs != 48000:
            raise ValueError('spectral loudness only supports wav files that are 48kHz, 16-bit')

        loudness_ch1 = Loudness(self.n_filter)
        loudness_ch2 = Loudness(self.n_filter)

        buffer_count = int((data.size/2)/self.buffer_size)
        for buffer in range(buffer_count):
            buffer_start = self.buffer_size * buffer
            buffer_stop = self.buffer_size * (buffer + 1)

            print(buffer_start, buffer_stop)

            if buffer is 0:
                pad = np.zeros((self.numtaps, 2))
                padded_stop = self.numtaps + buffer_stop + self.numtaps
                data_start = np.append(pad, data, 0)
                ch1 = data_start[: padded_stop, :1]
                ch2 = data_start[: padded_stop, 1:]

            elif buffer == buffer_count - 1:
                break
            else:
                padded_start = buffer_start - self.numtaps
                padded_stop = buffer_stop + self.numtaps
                ch1 = data[padded_start: padded_stop, :1]
                ch2 = data[padded_start: padded_stop, 1:]

            bpfb = BandpassFilterBank(n_filter=self.n_filter, numtaps=self.numtaps)

            freq_values = bpfb.corner_freq
            filtered_array_ch1 = self.remove_padding(bpfb.filter_bank(np.ndarray.flatten(ch1)))
            filtered_array_ch2 = self.remove_padding(bpfb.filter_bank(np.ndarray.flatten(ch2)))

            momentary_ch1, short_term_ch1, true_peak_ch1, dynamic_range_ch1 = loudness_ch1.process(filtered_array_ch1)
            momentary_ch2, short_term_ch2, true_peak_ch2, dynamic_range_ch2 = loudness_ch2.process(filtered_array_ch2)

            print('Dynamic Range: {} {}'.format(dynamic_range_ch1, dynamic_range_ch2))


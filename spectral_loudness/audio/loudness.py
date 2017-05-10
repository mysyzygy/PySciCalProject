import scipy.signal as signal
import numpy as np


class Fifo(object):
    def __init__(self, fifo_columns, fifo_row=30):
        self.fifo_row = fifo_row
        self.fifo_shape = (self.fifo_row, fifo_columns)
        self.fifo = np.empty(self.fifo_shape, dtype='float64')
        self.fifo.fill(-96)

    def set_fifo(self, input_buffer):
        for i in range(len(self.fifo) - 1):
            self.fifo[i:i+1] = self.fifo[i+1:i+2]
        self.fifo[-1] = input_buffer

    def get_fifo_segment(self, range):
        segment = self.fifo[range:]
        return segment


class Loudness(object):
    def __init__(self, n_filter):
        self.n_filter = n_filter
        self.momentary_loudness_fifo = [Fifo((4800))] * self.n_filter
        self.short_term_loudness_fifo = [Fifo((4800))] * self.n_filter
        self.momentary_loudness_value = None
        self.short_term_loudness_value = None
        self.true_peak_value = None

    def normalize(self, buffer):
        return buffer / (2. ** 15)

    @staticmethod
    def k_weight(buffer):
        # perform K-weight shelf filter
        a_shelf = np.array([1.0, -1.69065929318241, 0.73248077421585])
        b_shelf = np.array([1.53512485958697, -2.6916918940638, 1.19839281085285])
        zi = signal.lfilter_zi(b_shelf, a_shelf)
        z_shelf, _ = signal.lfilter(b_shelf, a_shelf, buffer, zi=zi * buffer[0])

        # perform K-weight highpass filter
        a_highpass = np.array([1.0, -1.99004745483398, 0.99007225036621])
        b_highpass = np.array([1.0, -2.0, 1.0])
        zi = signal.lfilter_zi(b_shelf, a_shelf)
        k_weight_buf, _ = signal.lfilter(b_highpass, a_highpass, buffer, zi=zi * z_shelf[0])

        return k_weight_buf

    @staticmethod
    def mean_square(buffer):
        mean_square = np.sum(buffer ** 2) / len(buffer)
        return mean_square

    @staticmethod
    def lufs(buffer):
        log_buf = 10 * np.log10(buffer)
        lufs_buf = -0.691 + log_buf
        return lufs_buf

    @staticmethod
    def convert_fs_to_db(input_value):
        return 20 * np.log10(input_value * 0.5 + 0.5)

    @staticmethod
    def convert_db_to_fs(input_value):
        return (10 ** (input_value / 20)) * 2 - 0.5

    @staticmethod
    def true_peak(buffer):
        coeffs = np.array([0.0017089843750,
                           -0.0291748046875,
                           -0.0189208984375,
                           -0.0083007812500,
                           0.0109863281250,
                           0.0292968750000,
                           0.0330810546875,
                           0.0148925781250,
                           -0.0196533203125,
                           -0.0517578125000,
                           -0.0582275390625,
                           -0.0266113281250,
                           0.0332031250000,
                           0.0891113281250,
                           0.1015625000000,
                           0.0476074218750,
                           -0.0594482421875,
                           -0.1665039062500,
                           -0.2003173828125,
                           -0.1022949218750,
                           0.1373291015625,
                           0.4650878906250,
                           0.7797851562500,
                           0.9721679687500,
                           0.9721679687500,
                           0.7797851562500,
                           0.4650878906250,
                           0.1373291015625,
                           -0.1022949218750,
                           -0.2003173828125,
                           -0.1665039062500,
                           -0.0594482421875,
                           0.0476074218750,
                           0.1015625000000,
                           0.0891113281250,
                           0.0332031250000,
                           -0.0266113281250,
                           -0.0582275390625,
                           -0.0517578125000,
                           -0.0196533203125,
                           0.0148925781250,
                           0.0330810546875,
                           0.0292968750000,
                           0.0109863281250,
                           -0.0083007812500,
                           -0.0189208984375,
                           -0.0291748046875,
                           0.0017089843750])

        # TODO: added 12.04 attenuation
        true_peak_result = round(max(20 * np.log10(abs(signal.resample_poly(buffer, 4, 4, window=coeffs)))), 2)
        return true_peak_result

    def momentary_loudness(self, buffer, freq):
        self.momentary_loudness_fifo[freq].set_fifo(buffer)
        momentary_buffer = self.momentary_loudness_fifo[freq].get_fifo_segment(-4)
        k_weight_result = self.k_weight(momentary_buffer.flatten())
        mean_square_result = self.mean_square(k_weight_result)
        momentary_loudness_result = self.lufs(mean_square_result)
        return momentary_loudness_result

    def short_term_loudness(self, buffer, freq):
        self.short_term_loudness_fifo[freq].set_fifo(buffer)
        momentary_buffer = self.short_term_loudness_fifo[freq].get_fifo_segment(0)
        k_weight_result = self.k_weight(momentary_buffer.flatten())
        mean_square_result = self.mean_square(k_weight_result)
        momentary_loudness_result = self.lufs(mean_square_result)
        return momentary_loudness_result

    def process(self, input_buffer):
        input_buffer_norm = self.normalize(input_buffer)

        momentary_loudness_result = np.zeros((31, 1), dtype='float64')
        short_term_loudness_result = np.zeros((31, 1), dtype='float64')
        true_peak_result = np.zeros((31, 1), dtype='float64')
        dynamic_range = np.zeros((31, 1), dtype='float64')

        for freq, buffer in enumerate(input_buffer_norm):
            self.momentary_loudness_value = self.momentary_loudness(buffer, freq)
            self.short_term_loudness_value = self.short_term_loudness(buffer, freq)
            self.true_peak_value = self.true_peak(buffer)
            momentary_loudness_result[freq:] = self.momentary_loudness_value
            short_term_loudness_result[freq:] = self.short_term_loudness_value
            true_peak_result[freq:] = self.true_peak_value
            dynamic_range[freq:] = self.true_peak_value - (
                                    self.momentary_loudness_value + self.short_term_loudness_value) / 2
        return momentary_loudness_result, short_term_loudness_result, true_peak_result, dynamic_range


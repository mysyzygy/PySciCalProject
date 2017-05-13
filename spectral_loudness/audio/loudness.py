import scipy.signal as signal
import numpy as np


class Fifo(object):
    def __init__(self, fifo_freq=31, fifo_buffer=30, fifo_samples=4800,):
        self.fifo_buffer = fifo_buffer
        self.fifo_freq = fifo_freq
        self.fifo_samples = fifo_samples
        self.fifo_shape = (self.fifo_freq, self.fifo_buffer,  self.fifo_samples)
        self.fifo = np.empty(self.fifo_shape, dtype='float64')

    def set_fifo(self, freq,  input_buffer):
        for i in range(self.fifo_buffer - 1):
            self.fifo[freq, i] = self.fifo[freq, i+1]
        self.fifo[freq, -1] = input_buffer
        pass

    def get_fifo_segment(self, freq, window):
        segment = self.fifo[freq, window:]
        return segment


class Loudness(object):
    def __init__(self, n_filter, window=4800):
        self.n_filter = n_filter
        self.window = window
        self.loudness_fifo = Fifo(fifo_freq=self.n_filter)

        self.momentary_loudness_result = np.zeros((self.n_filter, 1), dtype='float64')
        self.momentary_loudness_result.fill(-96.)

        self.short_term_loudness_result = np.zeros((self.n_filter, 1), dtype='float64')
        self.short_term_loudness_result.fill(-96.)

        self.momentary_true_peak_result = np.zeros((self.n_filter, 1), dtype='float64')
        self.momentary_true_peak_result.fill(-96.)

        self.short_term_true_peak_result = np.zeros((self.n_filter, 1), dtype='float64')
        self.short_term_true_peak_result.fill(-96.)

        self.momentary_dynamic_range = np.zeros((self.n_filter, 1), dtype='float64')
        self.short_term_dynamic_range = np.zeros((self.n_filter, 1), dtype='float64')

        self.momentary_loudness_value = None
        self.short_term_loudness_value = None
        self.momentary_true_peak_value = None
        self.short_term_true_peak_value = None

    @staticmethod
    def normalize(buffer):
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

    def true_peak(self, buffer):
        coeffs = np.array([0.0017089843750, -0.0291748046875, -0.0189208984375, -0.0083007812500,
                           0.0109863281250,  0.0292968750000,  0.0330810546875,  0.0148925781250,
                          -0.0196533203125, -0.0517578125000, -0.0582275390625, -0.0266113281250,
                           0.0332031250000,  0.0891113281250,  0.1015625000000,  0.0476074218750,
                          -0.0594482421875, -0.1665039062500, -0.2003173828125, -0.1022949218750,
                           0.1373291015625,  0.4650878906250,  0.7797851562500,  0.9721679687500,
                           0.9721679687500,  0.7797851562500,  0.4650878906250,  0.1373291015625,
                          -0.1022949218750, -0.2003173828125, -0.1665039062500, -0.0594482421875,
                           0.0476074218750,  0.1015625000000,  0.0891113281250,  0.0332031250000,
                          -0.0266113281250, -0.0582275390625, -0.0517578125000, -0.0196533203125,
                           0.0148925781250,  0.0330810546875,  0.0292968750000,  0.0109863281250,
                          -0.0083007812500, -0.0189208984375, -0.0291748046875,  0.0017089843750])

        true_peak_result = round(max(20 * np.log10(abs(self.normalize(
                                     signal.resample_poly(buffer, 4, 4, window=coeffs))))), 2)
        return true_peak_result

    def momentary_loudness(self, freq):
        # window over 400ms
        momentary_buffer = self.loudness_fifo.get_fifo_segment(freq, -4)

        # apply k-weight filter
        k_weight_result = self.k_weight(momentary_buffer.flatten())

        # normalize from 16 bit to full scale
        k_weight_norm = self.normalize(k_weight_result)

        # calculate mean square
        mean_square_result = self.mean_square(k_weight_norm)

        # calculate momentary lufs
        momentary_loudness_result = self.lufs(mean_square_result)
        return momentary_loudness_result

    def short_term_loudness(self, freq):
        # window over 3s
        short_term_buffer = self.loudness_fifo.get_fifo_segment(freq, 0)

        # apply k-weight filter
        k_weight_result = self.k_weight(short_term_buffer.flatten())

        # normalize from 16 bit to full scale
        k_weight_norm = self.normalize(k_weight_result)

        # calculate mean square
        mean_square_result = self.mean_square(k_weight_norm)

        # calculate short term lufs
        momentary_loudness_result = self.lufs(mean_square_result)
        return momentary_loudness_result

    def process(self, input_buffer):
        for freq, buffer in enumerate(input_buffer):
            # set fifo with freq buffer
            self.loudness_fifo.set_fifo(freq, buffer)

            # calculate momentary loudness lufs
            self.momentary_loudness_value = self.momentary_loudness(freq)
            self.momentary_loudness_result[freq:] = self.momentary_loudness_value

            # find true peak over 400ms window
            self.momentary_true_peak_value = self.true_peak(self.loudness_fifo.get_fifo_segment(freq, -4).flatten())
            self.momentary_true_peak_result[freq:] = self.momentary_true_peak_value

            # calculate momentary dynamic range
            self.momentary_dynamic_range[freq:] = self.momentary_true_peak_value - self.momentary_loudness_value

            # calculate short term loudness lufs
            self.short_term_loudness_value = self.short_term_loudness(freq)
            self.short_term_loudness_result[freq:] = self.short_term_loudness_value

            # find true peak over 400ms window
            self.short_term_true_peak_value = self.true_peak(self.loudness_fifo.get_fifo_segment(freq, 0).flatten())
            self.short_term_true_peak_result[freq:] = self.short_term_true_peak_value

            # calculate momentary dynamic range
            self.short_term_dynamic_range[freq:] = self.short_term_true_peak_value - self.short_term_loudness_value

        return (self.momentary_loudness_result, self.short_term_loudness_result,
                self.momentary_true_peak_result, self.short_term_true_peak_result,
                self.momentary_dynamic_range, self.short_term_dynamic_range)

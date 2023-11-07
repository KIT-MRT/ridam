import numpy as np
from scipy import signal


def channels_to_complex(radar_frame):
    return np.vectorize(complex)(radar_frame[0], radar_frame[1])


def complex_to_channels(radar_frame, axis=0):
    return np.stack((radar_frame.real, radar_frame.imag), axis=axis).astype(float)


def rds(radar_samples, complex_radar_frame=False):
        if not complex_radar_frame:
             complex_radar_samples = channels_to_complex(radar_samples)
        else:
             complex_radar_samples = radar_samples
        
        window_range = np.tile(np.hanning(complex_radar_samples.shape[0]).reshape(-1, 1), (1, complex_radar_samples.shape[1]))
        window_velocity = np.tile(np.hanning(complex_radar_samples.shape[1]), (complex_radar_samples.shape[0], 1))

        rp = np.fft.fft(window_range * complex_radar_samples, axis=0)
        rds = np.fft.fft(window_velocity * rp, axis=1)
        rds = np.fft.fftshift(rds, axes=(1,))

        return rds


class ca_cfar_2d():
    def __init__(self, win_param=[16, 16, 8, 8], threshold=0.0):
        
        win_width = win_param[0]
        win_height = win_param[1]
        guard_width = win_param[2]
        guard_height = win_param[3]

        self.mask = np.ones((2 * win_height + 1, 2 * win_width + 1), dtype=bool)
        self.mask[win_height - guard_height:win_height + 1 + guard_height, win_width - guard_width:win_width + 1 + guard_width] = 0
        self.threshold = 10 ** (threshold / 10)
        self.num_valid_cells_in_window = signal.convolve2d(np.ones([192,64], dtype=float), self.mask, mode='same')

    def __call__(self, rds):
        rd_windowed_sum = signal.convolve2d(rds, self.mask, mode='same')
        rd_avg_noise_power = rd_windowed_sum / self.num_valid_cells_in_window
        rd_snr = rds / rd_avg_noise_power
        hit_matrix = rd_snr > self.threshold

        return hit_matrix

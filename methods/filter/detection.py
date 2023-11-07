import numpy as np
import scipy


class VariationDetector:
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, radar_frames):
        total_detections = np.zeros(shape=[radar_frames.shape[0], radar_frames.shape[2], radar_frames.shape[3]])

        for idx, radar_frame in enumerate(radar_frames):
            radar_img = np.abs(np.vectorize(complex)(radar_frame[0], radar_frame[1]))
            thresholds = np.mean(radar_img, axis=0)
            detections = np.ones_like(radar_img, dtype=int)
    
            filter = np.array([
                [0, 1 , 0],
                [0, 0 , 0],
                [0, -1 , 0]
            ])
            diff_img = scipy.ndimage.convolve(radar_img, filter)

            for j, threshold in enumerate(thresholds):
                detections[:, j] = diff_img[:, j] > threshold

            total_detections[idx] = np.ones_like(radar_img, dtype=int) - detections

        return total_detections


class LaplacianDetector():
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, radar_frames):
        total_detections = np.zeros(shape=[radar_frames.shape[0], radar_frames.shape[2], radar_frames.shape[3]])

        for idx, radar_frame in enumerate(radar_frames):
            radar_img = np.abs(np.vectorize(complex)(radar_frame[0], radar_frame[1]))
            thresholds = np.mean(radar_img, axis=0)
            detections = np.ones_like(radar_img, dtype=int)
    
            diff_img = scipy.ndimage.laplace(radar_img)

            for j, threshold in enumerate(thresholds):
                detections[:, j] = diff_img[:, j] > threshold

            total_detections[idx] = np.ones_like(radar_img, dtype=int) - detections

        return total_detections

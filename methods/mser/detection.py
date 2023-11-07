import cv2
import numpy as np


class MSER():
    def __init__(self) -> None:
        self.mser = cv2.MSER_create(delta=5, min_area=2, max_area=20)
        
    def __call__(self, radar_frames):
        total_detections = np.ones(shape=[radar_frames.shape[0], radar_frames.shape[2], radar_frames.shape[3]])

        for idx, radar_frame in enumerate(radar_frames):
            radar_frame = np.vectorize(complex)(radar_frame[0], radar_frame[1])

            # fix log of 0
            radar_frame[radar_frame == 0] = 0.000000000000001

            radar_img = np.log10(np.abs(radar_frame))
            radar_img =(
                (radar_img - radar_img.min()) * 
                (1/(radar_img.max() - radar_img.min()) * 255)
            ).astype('uint8')


            mask = np.ones((radar_img.shape[0], radar_img.shape[1]), dtype=np.uint8)
            regions, _ = self.mser.detectRegions(radar_img)
            hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
            cv2.polylines(mask, hulls, 1, (0))

            total_detections[idx] = mask

        return total_detections

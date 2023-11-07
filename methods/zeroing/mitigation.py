import numpy as np
import torch

class Zeroing():
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, radar_frames, masks):
        radar_frames = radar_frames.cpu().numpy()
        masks = masks.cpu().numpy()

        mitigations = np.zeros_like(radar_frames)

        for idx in range(len(radar_frames)):
            radar_frame = radar_frames[idx]
            mask = masks[idx]

            mitigations[idx] = radar_frame * mask

        return torch.tensor(mitigations)

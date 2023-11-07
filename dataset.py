import numpy as np
import torch
from torch.utils.data import Dataset

from detection import rds, complex_to_channels


class AugmentedRadarFrameDataset(Dataset):
    def __init__(self, radar_frames, interference_simulator, norm, is_rds=False, db=None, db_threshold=5.0):
        self.radar_frames = radar_frames
        self.interference_simulator = interference_simulator
        self.is_rds = is_rds
        self.norm = norm
        self.db = db
        self.db_threshold = db_threshold

    def __len__(self):
        return len(self.radar_frames)

    def __getitem__(self, idx):
        clean, mask, disturbed  = self.interference_simulator.simulate(self.radar_frames[idx])

        if self.is_rds:
            return \
                torch.tensor(complex_to_channels(self.norm(rds(clean, complex_radar_frame=True))).astype(np.float32)), \
                torch.tensor(mask, dtype=torch.long), \
                torch.tensor(complex_to_channels(self.norm(rds(disturbed, complex_radar_frame=True))).astype(np.float32))
        
        clean = torch.tensor(complex_to_channels(clean), dtype=torch.float32)
        mask  = torch.tensor(mask, dtype=torch.long)
        disturbed = torch.tensor(complex_to_channels(disturbed), dtype=torch.float32)

        return self.norm(clean), mask, self.norm(disturbed)


class StaticRadarFrameDataset(Dataset):
    def __init__(self, clean, mask, disturbed, norm, is_rds=False):
        assert len(clean) == len(mask)
        assert len(mask) == len(disturbed)

        self.clean = clean
        self.mask = mask
        self.disturbed = disturbed
        self.norm = norm
        self.is_rds = is_rds

    def __len__(self):
        return len(self.clean)

    def __getitem__(self, idx):
        clean = self.clean[idx]
        mask = self.mask[idx]
        disturbed = self.disturbed[idx]

        if self.is_rds:
            return \
                torch.tensor(complex_to_channels(self.norm(rds(clean, complex_radar_frame=True))).astype(np.float32)), \
                torch.tensor(mask, dtype=torch.long), \
                torch.tensor(complex_to_channels(self.norm(rds(disturbed, complex_radar_frame=True))).astype(np.float32))

        if self.norm:
            return self.norm(clean), mask, self.norm(disturbed)
        else:
            return clean, mask, disturbed

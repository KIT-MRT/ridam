import argparse

import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import AugmentedRadarFrameDataset
from simulation import InterferenceSimulator


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("clean_radar_frames", help="path to clean radar frames (generated by radical/extract.py)")
    parser.add_argument("output_path", help="output path to store train, val and test split.")
    args = parser.parse_args()

    radar_frames = np.load(args.clean_radar_frames)
    permutate_idx = np.arange(len(radar_frames))
    np.random.shuffle(permutate_idx)
    radar_frames = radar_frames[permutate_idx]
    
    len_train = int(0.8*len(radar_frames))
    len_val = len(radar_frames) - len_train

    radar_frames_train = radar_frames[:len_train]
    radar_frames_val = radar_frames[len_train:len_train+len_val]

    norm = lambda x: x

    batch_size = 64
    interference_simulator = InterferenceSimulator.default(min_radar=0, max_radar=4)
    val_data = AugmentedRadarFrameDataset(radar_frames_val, interference_simulator, norm=norm, db=None, db_threshold=2)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=32, pin_memory=True)

    val_clean = np.zeros(shape=(len_val, 2, 192, 64))
    val_mask = np.zeros(shape=(len_val, 192, 64), dtype=int)
    val_disturbed = np.zeros(shape=(len_val, 2, 192, 64))
    for i, data in enumerate(tqdm(val_dataloader)):
        val_clean[i*batch_size:(i)*batch_size + data[0].shape[0]] = data[0]
        val_mask[i*batch_size:(i)*batch_size + data[0].shape[0]] = data[1]
        val_disturbed[i*batch_size:(i)*batch_size + data[0].shape[0]] = data[2]

    val_clean = np.vectorize(complex)(val_clean[:,0,:,:], val_clean[:,1,:,:])
    val_disturbed = np.vectorize(complex)(val_disturbed[:,0,:,:], val_disturbed[:,1,:,:])

    np.save(f"{args.output_path}/train_clean.npy", radar_frames_train)
    np.save(f"{args.output_path}/val_clean.npy", val_clean)
    np.save(f"{args.output_path}/val_mask.npy", val_mask)
    np.save(f"{args.output_path}/val_disturbed.npy", val_disturbed)

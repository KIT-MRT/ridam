import argparse
import glob

import numpy as np
from tqdm import tqdm
from radicalsdk.h5dataset import H5DatasetLoader
from radicalsdk.radar.config_v1 import read_radar_params
from radicalsdk.radar.v1 import RadarFrame


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path", help="dataset path, for example '/path/to/radical/radar_30m/*.bag.h5'")
    parser.add_argument("config_path", help="val_clean")
    parser.add_argument("output_file", help="val_clean")
    args = parser.parse_args()

    # these parameters are based on radical config
    samples = 192
    frame_size = 64
    clean = []
    for hd5_file_path in glob.glob(args.dataset_path):
        print(f"Reading {hd5_file_path}")
        data = H5DatasetLoader(hd5_file_path, ['radar'])
        radar_config = read_radar_params(args.config_path)
        rf = RadarFrame(radar_config)

        radar_frames = np.ndarray(shape=[len(data), samples, frame_size], dtype=np.complex64)
        for idx, raw_data in enumerate(tqdm(data)):
            raw_radar = raw_data[0]
            raw_radar = np.transpose(raw_radar, (2, 0, 1))
            radar_frames[idx] = raw_radar[:,:,0]

        clean.append(radar_frames)

    total_len = 0
    for c in clean:
        total_len += c.shape[0]
    print(f"{total_len} frames")

    seen = 0
    total_radar_frames = np.ndarray(shape=[total_len, samples, frame_size], dtype=np.complex64)
    for c in clean:
        total_radar_frames[seen:seen+c.shape[0]] = c
        seen += c.shape[0]
    
    # Unfortunaetly some radar frames are only filled with 0
    bad_idx = []
    for idx, frame in enumerate(total_radar_frames):
        if np.abs(frame).mean() == 0:
            bad_idx.append(idx)
    print(f"bad frames: {len(bad_idx)}")
    total_radar_frames = np.delete(total_radar_frames, bad_idx, axis=0)

    print(f"radar frames after removel of zero frames: {len(total_radar_frames)}")
    np.save(args.output_file, total_radar_frames)

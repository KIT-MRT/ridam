# End-to-End Trainable Deep Neural Network for Radar Interference Detection and Mitigation 

## Preparation

1: Download the RaDICaL dataset and install the RaDICaL SDK https://github.com/moodoki/radical_sdk

2: Extract .h5 bags to numpy files using `radical/extract.py /path/to/radical/radar_30m/*.bag.h5 /path/radical/radarcfg/outdoor_human_rcs_30m.cfg /path/to/output/radar_frames.npy`.

3: Generate a train, val, and test split using `python generate_training_data.py /path/to/output/radar_frames.npy /path/to/generated/files`.


## Training

First set required envorinment variables.

```bash
export NORM_PATH=/path/to/norm/pkl
export TRAIN_CLEAN=/path/to/generated/files/train_clean.npy
export VAL_CLEAN=/path/to/generated/files/val_clean.npy
export VAL_MASK=/path/to/generated/files/val_mask.npy
export VAL_DISTURBED=/path/to/generated/files/val_disturbed.npy
```

Depending on the method you want to train you can use different training scripts:

RIDAM Detection and Mitigation
- `python train_ridam_detection_mitigation.py $NORM_PATH $TRAIN_CLEAN $VAL_CLEAN $VAL_MASK $VAL_DISTURBED`

AE-Gate Detection
- `python train_ae-gate_mitigation.py $NORM_PATH $TRAIN_CLEAN $VAL_CLEAN $VAL_MASK $VAL_DISTURBED`

AE-Gate Mitigation
- `python train_ae-gate_detection.py $NORM_PATH $TRAIN_CLEAN $VAL_CLEAN $VAL_MASK $VAL_DISTURBED`

CNNTD Mitigation
- `python train_cnntd_mitigation.py $NORM_PATH $TRAIN_CLEAN $VAL_CLEAN $VAL_MASK $VAL_DISTURBED`

CNNRD Mitigation
- `python train_cnnrd_mitigation.py $NORM_PATH $TRAIN_CLEAN $VAL_CLEAN $VAL_MASK $VAL_DISTURBED`

## Evaluation

Generate tests sets using `generate_training_data.py`. Use val_clean.npy as clean input and change simulation parameters according to your desired testing environment.

### Detection Evaluation & Mitigation

To evaluate the detection and mitigation capability of different methods use the `evaluate_detection.py` and `evaluate_mitigation.py`, respectively. 
Inside you can comment/uncomment methods you want to evaluate.
Please make sure that you set the ckpt path for the learned methods (commented by todo).

## Checkpoints

will be released soon.
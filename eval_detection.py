import argparse
import pickle

import torch
from torch.utils.data import DataLoader
import numpy as np

from dataset import StaticRadarFrameDataset
from methods.ae_gate.detection import DetectionCNN, LitDetectionCNN
from methods.filter.detection import VariationDetector, LaplacianDetector
from methods.mser.detection import MSER
from metrics import Metrics, evaluate_detection
from methods.ridam import LitRadarInterferenceDetectionAndMitigation, RadarInterferenceDetectionAndMitigation


if __name__ == "__main__":
    # How to use:
    #   Pass test clean, mask, disturbed data
    #   Uncomment methods and norm that you want to evaluate
    #       Change path to ckpt    
    #   Run Code

    parser = argparse.ArgumentParser()
    parser.add_argument("norm", help="norm")
    parser.add_argument("test_clean", help="val_clean")
    parser.add_argument("test_mask", help="val_mask")
    parser.add_argument("test_disturbed", help="val_disturbed")
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA Unavailable"
    device = torch.device('cuda:0')

    with open(args.norm, 'rb') as f:
        norm = pickle.load(f)

    clean = np.load(args.val_clean)
    mask = np.load(args.val_mask)
    disturbed = np.load(args.val_disturbed)
    test_data = StaticRadarFrameDataset(clean, mask, disturbed, norm=norm, is_rds=True)
    test_dataloader = DataLoader(test_data, batch_size=32, shuffle=False)


    # AE-Gate
    # model_aegate_detect = DetectionCNN().to(device)
    # model_aegate_detect = LitDetectionCNN(model_aegate_detect, lr=0)
    # checkpoint = torch.load("todo: change to cktp path")
    # model_aegate_detect.load_state_dict(checkpoint["state_dict"])
    # model_aegate_detect = model_aegate_detect.det
    # model_aegate_detect.eval()

    # # RIDAM
    # model_ridam = RadarInterferenceDetectionAndMitigation(base_dim=8).to(device)
    # model_ridam = LitRadarInterferenceDetectionAndMitigation(model_ridam, 0, 0)
    # checkpoint = torch.load("todo: change to cktp path")
    # model_ridam.load_state_dict(checkpoint["state_dict"])
    # model_ridam = model_ridam.rim
    # model_ridam.eval()

    # Variation
    # model_variation = VariationDetector()
    
    # LD
    # model_ld = LaplacianDetector()

    # MSER
    # model_mser = MSER()


    metrics_variation = Metrics()
    metrics_ld = Metrics()
    metrics_mser = Metrics()
    metrics_ridam = Metrics()
    metrics_ae_gate_detect = Metrics()
    with torch.no_grad():
        for i, data in enumerate(test_dataloader):
            clean, mask, disturbed = data[0].to(device), data[1].to(device), data[2].to(device)
            # MSER
            # mask_mser = model_mser(disturbed.cpu().numpy())
            # metrics_mser = evaluate_detection(
            #     mask.cpu().numpy().flatten().tolist(),
            #     model_mser(disturbed.cpu().numpy()).flatten().tolist(),
            #     metrics_mser
            # )

            # LaplaceDetector
            # metrics_ld = evaluate_detection(
            #     mask.cpu().numpy().flatten().tolist(),
            #     model_ld(disturbed.cpu().numpy()).flatten().tolist(),
            #     metrics_ld
            # )
            
            # Variation
            # mask_variation = model_variation(disturbed.cpu().numpy())
            # metrics_variation = evaluate_detection(
            #     mask.cpu().numpy().flatten().tolist(),
            #     mask_variation.flatten().tolist(),
            #     metrics_variation
            # )

            # RIDAM
            # mask_ridam, _ = model_ridam(disturbed)
            # mask_ridam = torch.argmax(torch.nn.functional.softmax(mask_ridam, dim=1), dim=1)
            # metrics_ridam = evaluate_detection(
            #      mask.cpu().numpy().flatten().tolist(),
            #      mask_ridam.cpu().numpy().flatten().tolist(),
            #      metrics_ridam
            # )

            # AE-GATED
            # mask_ae = model_aegate_detect(disturbed)
            # mask_ae = torch.argmax(torch.nn.functional.softmax(mask_ae, dim=1), dim=1)
            # metrics_ae_gate_detect = evaluate_detection(
            #      mask.cpu().numpy().flatten().tolist(),
            #      mask_ae.cpu().numpy().flatten().tolist(),
            #      metrics_ae_gate_detect
            # )
    
    # P, R, F1 = metrics_mser.f1_score()
    # print(F"MSER: {round(P, 4)} & {round(R, 4)} & {round(F1, 4)}")
    # P, R, F1 = metrics_variation.f1_score()
    # print(F"Var: {round(P, 4)} & {round(R, 4)} & {round(F1, 4)}")
    # P, R, F1 = metrics_ld.f1_score()
    # print(F"LD: {round(P, 4)} & {round(R, 4)} & {round(F1, 4)}")
    # P, R, F1 = metrics_ae_gate_detect.f1_score()
    # print(F"GA: {round(P, 4)} & {round(R, 4)} & {round(F1, 4)}")
    # P, R, F1 = metrics_ridam.f1_score()
    # print(F"RIDAM: {round(P, 4)} & {round(R, 4)} & {round(F1, 4)}")

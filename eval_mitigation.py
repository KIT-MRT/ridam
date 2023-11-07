import argparse
import numpy as np
import pickle
import torch
from torch.utils.data import DataLoader

from methods.ae_gate.detection import DetectionCNN, LitDetectionCNN
from methods.ae_gate.mititgation import AEGateMitigation, LitAEGateMitigation
from methods.ar.mitigation import ArMitigation
from methods.cnnrd.mitigation import CnnRdMitigation, ComplexNorm, LitCnnRdMitigation
from methods.cnntd.mitigation import CnnTdMitigation, LitCnnTdMitigation
from methods.ridam import RadarInterferenceDetectionAndMitigation, LitRadarInterferenceDetectionAndMitigation
from dataset import StaticRadarFrameDataset
from methods.zeroing.mitigation import Zeroing
from metrics import Metrics, evaluate_mitigation

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

    # with open("path_to_norm.pkl", 'rb') as f:
        # norm = pickle.load(f)
    # norm = ComplexNorm(
    #     -2.843303e-07,
    #     1.8715082e-07,
    #     8042.9546,
    #     8700.103
    # )

    clean = np.load(args.val_clean)
    mask = np.load(args.val_mask)
    disturbed = np.load(args.val_disturbed)
    test_data = StaticRadarFrameDataset(clean, mask, disturbed, norm=norm, is_rds=True)
    test_dataloader = DataLoader(test_data, batch_size=32, shuffle=False)

    # Zeroing
    # zeroing = Zeroing()

    # Ar (requires matlab)
    # ar = ArMitigation()

    # RIDAM
    # model_ridam = RadarInterferenceDetectionAndMitigation(base_dim=8).to(device)
    # model_ridam = LitRadarInterferenceDetectionAndMitigation(model_ridam, 0, 0)
    # checkpoint = torch.load("todo: change to cktp path")
    # model_ridam.load_state_dict(checkpoint["state_dict"])
    # model_ridam = model_ridam.rim
    # model_ridam.eval()

    # CNNTD
    # model_cnntd = CnnTdMitigation().to(device)
    # model_cnntd = LitCnnTdMitigation(model_cnntd, 0)
    # checkpoint = torch.load("todo: change to cktp path")
    # model_cnntd.load_state_dict(checkpoint["state_dict"])
    # model_cnntd = model_cnntd.mit
    # model_cnntd.eval()

    # CNNRD
    # model_cnnrd = CnnRdMitigation(num_conv_layer=4, num_filters=16, filter_size=(3,3), input_size=(2, 192, 64)).to(device)
    # model_cnnrd = LitCnnRdMitigation(model_cnnrd, 0)
    # checkpoint = torch.load("todo: change to cktp path")
    # model_cnnrd.load_state_dict(checkpoint["state_dict"])
    # model_cnnrd = model_cnnrd.mit
    # model_cnnrd.eval()

    # AE-GATE
    # detection
    # model_aegate_detect = DetectionCNN().to(device)
    # model_aegate_detect = LitDetectionCNN(model_aegate_detect, lr=0)
    # checkpoint = torch.load("todo: change to cktp path")
    # model_aegate_detect.load_state_dict(checkpoint["state_dict"])
    # model_aegate_detect = model_aegate_detect.det
    # model_aegate_detect.eval()

    # mitigation
    # model_aegate_mitigate = AEGateMitigation().to(device)
    # model_aegate_mitigate = LitAEGateMitigation(model_aegate_mitigate, 0.0)
    # checkpoint = torch.load("todo: change to cktp path")
    # model_aegate_mitigate.load_state_dict(checkpoint["state_dict"])
    # model_aegate_mitigate = model_aegate_mitigate.mit
    # model_aegate_mitigate.eval()

    #
    # Eval
    #
    metrics_ridam = Metrics()
    metrics_aegate = Metrics()
    metrics_cnntd = Metrics()
    metrics_cnnrd = Metrics()
    metrics_disturbed = Metrics()
    metrics_zeroing = Metrics()
    metrics_ar = Metrics()
    with torch.no_grad():
        for i, data in enumerate(test_dataloader, 0):
            clean, mask, disturbed = data[0].to(device), data[1].to(device), data[2].to(device)

            # zeroing
            # mitigated_zeroing = zeroing(disturbed, mask)
            # metrics_zeroing = evaluate_mitigation(
            #      clean,
            #      disturbed,
            #      mitigated_zeroing,
            #      metrics_zeroing
            # )

            # AR
            # mitigated_ar = ar(disturbed, mask)
            # metrics_ar = evaluate_mitigation(
            #     clean,
            #     disturbed,
            #     mitigated_ar,
            #     metrics_ar
            # )

            # _, mitigated_ridam = model_ridam(disturbed, mask)
            # metrics_ridam = evaluate_mitigation(
            #     clean,
            #     disturbed,
            #     mitigated_ridam,
            #     metrics_ridam,
            #     is_rds=False
            # )

            # CNNTD
            # mitigated_cnntd = model_cnntd(disturbed, mask)
            # metrics_cnntd = evaluate_mitigation(
            #     clean,
            #     disturbed,
            #     mitigated_cnntd,
            #     metrics_cnntd,
            #     is_rds=False
            # )

            # CNNRD
            # mitigated_cnnrd = model_cnnrd(disturbed)
            # metrics_cnnrd = evaluate_mitigation(
            #     clean,
            #     disturbed,
            #     mitigated_cnnrd,
            #     metrics_ridam,
            #     is_rds=True
            # )

            # AEGATE
            # detect
            # mask_ae = model_aegate_detect(disturbed)
            # mask_ae = torch.argmax(torch.nn.functional.softmax(mask_ae, dim=1), dim=1)
            # mitigate
            # mitigated_aegate = model_aegate_mitigate(disturbed, mask_ae)
            # metrics_aegate = evaluate_mitigation(
            #     clean,
            #     disturbed,
            #     mitigated_aegate,
            #     metrics_aegate,
            #     is_rds=False
            # )

            # metrics_disturbed = evaluate_mitigation(
            #     clean,
            #     disturbed,
            #     clean,
            #     metrics_disturbed
            # )
            
    # print(db)
    # P, R, F1 = metrics_disturbed.f1_score()
    # print(F"Clean: {round(np.array(metrics_disturbed.sinr_m).mean(), 5)} & {round(np.array(metrics_disturbed.evm_m).mean(), 5)} & {round(P, 4)} & {round(R, 4)} & {round(F1, 4)}") 
    # P, R, F1 = metrics_ar.f1_score()
    # print(F"AR: {round(np.nanmean(np.array(metrics_ar.sinr_m)), 4)} & {round(np.nanmean(np.array(metrics_ar.evm_m)), 4)} & {round(P, 4)} & {round(R, 4)} & {round(F1, 4)}")
    # P, R, F1 = metrics_zeroing.f1_score()
    # print(F"Zeroing: {round(np.nanmean(np.array(metrics_zeroing.sinr_m)), 4)} & {round(np.nanmean(np.array(metrics_zeroing.evm_m)), 4)} & {round(P, 4)} & {round(R, 4)} & {round(F1, 4)}")
    # P, R, F1 = metrics_ridam.f1_score()
    # print(F"RIDAM: {round(np.array(metrics_ridam.sinr_m).mean(), 5)} & {round(np.array(metrics_ridam.evm_m).mean(), 5)} & {round(P, 5)} & {round(R, 5)} & {round(F1, 5)}")
    # P, R, F1 = metrics_cnntd.f1_score()
    # print(F"CNNTD: {round(np.array(metrics_cnntd.sinr_m).mean(), 4)} & {round(np.array(metrics_cnntd.evm_m).mean(), 4)} & {round(P, 4)} & {round(R, 4)} & {round(F1, 4)}")
    # P, R, F1 = metrics_cnnrd.f1_score()
    # print(F"CNNRD: {round(np.array(metrics_cnnrd.sinr_m).mean(), 4)} & {round(np.array(metrics_cnnrd.evm_m).mean(), 4)} & {round(P, 4)} & {round(R, 4)} & {round(F1, 4)}")
    # P, R, F1 = metrics_aegate.f1_score()
    # print(F"AE-GATE: {round(np.array(metrics_aegate.sinr_m).mean(), 4)} & {round(np.array(metrics_aegate.evm_m).mean(), 4)} & {round(P, 4)} & {round(R, 4)} & {round(F1, 4)}")

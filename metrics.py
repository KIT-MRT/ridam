import numpy as np
from sklearn.metrics import confusion_matrix

from detection import ca_cfar_2d, rds


def channels_to_complex(radar_frame):
    return np.vectorize(complex)(radar_frame[0], radar_frame[1])


def calculate_f1_score(TP, FP, TN, FN):
    import math
    if math.isclose(TP + FP, 0.0):
        P = 0
    else:
        P = TP / (TP + FP)

    if math.isclose(TP + FN, 0.0):
        R = 0
    else:
        R = TP / (TP + FN)

    if math.isclose(P + R, 0.0):
        return 0

    return P, R, 2 * (P * R) / (P + R)


class Metrics:
    def __init__(self) -> None:
        self.tn = [] 
        self.fn = []
        self.fp = []
        self.tp = []
        self.sinr_c = []
        self.sinr_d = []
        self.sinr_m = []
        self.evm_d = []
        self.evm_m = []
        self.PPMSE_d = []
        self.PPMSE_m = []


    def f1_score(self):
        return calculate_f1_score(sum(self.tp), sum(self.fp), sum(self.tn), sum(self.fn))


def evaluate_detection(mask, mask_prediction, metrics):
    cm = confusion_matrix(mask, mask_prediction)
    metrics.tn.append(cm[0][0])
    metrics.fn.append(cm[1][0])
    metrics.fp.append(cm[0][1])
    metrics.tp.append(cm[1][1])

    return metrics
    

def sinr(rds, target_cells, noise_cells):
    return 10 * np.log10(
            np.mean(np.abs(rds[target_cells]) ** 2)
            /
            np.mean(np.abs(rds[noise_cells]) ** 2)
        )


def evm(rds_clean, rds_mitigated, target_cells):
    return np.mean(
        np.abs(rds_clean[target_cells] - rds_mitigated[target_cells]) / np.abs(rds_clean[target_cells])
    )


def evaluate_mitigation(clean, disturbed, mitigated, metrics, is_rds=False):
    from sklearn.metrics import confusion_matrix

    cfar = ca_cfar_2d(threshold=10.75)

        
    for idx in range(len(clean)):
        c = clean[idx].cpu().numpy()
        d = disturbed[idx].cpu().numpy()
        m = mitigated[idx].cpu().numpy()

        if is_rds:
            rds_clean = channels_to_complex(c)
            rds_disturbed = channels_to_complex(d)
            rds_mitigated = channels_to_complex(m)
        else:
            rds_clean = rds(c)
            rds_disturbed = rds(d)
            rds_mitigated = rds(m)

        c_map = cfar(np.abs(rds_clean)**2)
        m_map = cfar(np.abs(rds_mitigated)**2)

        target_idx = np.unravel_index(np.argwhere(c_map.flatten() == 1.0), c_map.shape)
        noise_idx = np.unravel_index(np.argwhere(c_map.flatten() == 0.0), c_map.shape)

        metrics.sinr_c.append(sinr(rds_clean, target_idx, noise_idx))
        metrics.sinr_d.append(sinr(rds_disturbed, target_idx, noise_idx))
        metrics.sinr_m.append(sinr(rds_mitigated, target_idx, noise_idx))

        metrics.evm_d.append(evm(rds_clean, rds_disturbed, target_idx))
        metrics.evm_m.append(evm(rds_clean, rds_mitigated, target_idx))

        cm = confusion_matrix(c_map.flatten().tolist(), m_map.flatten().tolist())
        metrics.tn.append(cm[0][0])
        metrics.fn.append(cm[1][0])
        metrics.fp.append(cm[0][1])
        metrics.tp.append(cm[1][1])

    return metrics

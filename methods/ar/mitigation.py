import numpy as np
import torch
import matlab.engine


class ArMitigation():
    def __init__(self):
        super().__init__()
        self.eng = matlab.engine.start_matlab()

    def __call__(self, x, mask):
        x = x.cpu().numpy()
        mask = mask.cpu().numpy()
        mask = np.tile(mask[:, np.newaxis, :, :], (1, 2, 1, 1))

        x = x * mask
        x[x == 0] = np.NaN

        output = np.ndarray(shape=[len(x), 2, 192, 64], dtype=np.float32)
        for idx in range(len(x)):
            matlab_input_real = matlab.double(x[idx, 0, :, :].tolist())
            matlab_input_imag = matlab.double(x[idx, 1, :, :].tolist())
            recovered_real = np.array(self.eng.fillgaps(matlab_input_real), dtype=np.float32)
            recovered_imag = np.array(self.eng.fillgaps(matlab_input_imag), dtype=np.float32)
            recovered = np.stack((recovered_real, recovered_imag), axis=0)

            output[idx] = recovered

        return torch.tensor(output)

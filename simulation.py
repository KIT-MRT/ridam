from dataclasses import dataclass
import math
import random
import copy

import numpy as np


@dataclass
class EgoRadar():
    fc: float
    bw: float
    time: float
    lpf_cutoff: float
    samples: int
    chirps: int

    def slope(self) -> float:
        return self.bw / self.time
    
    def time_per_sample(self) -> float:
        return (self.time) / self.samples


@dataclass
class InterferingRadar:
    fc: float
    bw: float
    chirps: int
    time: float
    power: float

    def slope(self) -> float:
        return self.bw / self.time


class InterferenceSimulator():
    def __init__(
        self,
        ego_radar,
        min_fc, 
        max_fc, 
        min_bw,
        max_bw,
        min_chirps,
        max_chirps,
        min_time, 
        max_time,
        min_power, 
        max_power,
        min_interfering_radars=1,
        max_interfering_radars=4
    ) -> None:
        super(InterferenceSimulator).__init__()
        self.ego_radar = ego_radar
        self.min_fc = min_fc
        self.max_fc = max_fc
        self.min_bw = min_bw
        self.max_bw = max_bw
        self.min_chirps = min_chirps
        self.max_chirps = max_chirps
        self.min_time = min_time
        self.max_time = max_time
        self.min_power = min_power
        self.max_power = max_power
        self.max_interfering_radars = max_interfering_radars
        self.min_interfering_radars = min_interfering_radars

    def default(min_radar=0, max_radar=4):
        return InterferenceSimulator(
            EgoRadar(77e09, 1150e06, 23.338e-6, 15e6, 192, 64),
            min_fc=77e09,
            max_fc=78e09,
            min_bw=100e06,
            max_bw=1000e06,
            min_chirps=16,
            max_chirps=512,
            min_time=15e-6,
            max_time=100e-6,
            min_power=300,
            max_power=600,
            min_interfering_radars=min_radar,
            max_interfering_radars=max_radar
        )
    
    
    def simulate(self, radar_frame):
        num_interferer = random.randint(self.min_interfering_radars, self.max_interfering_radars)
        interfering_radars = [self.gen_interfering_radar() for _ in range(num_interferer)]

        interference = np.zeros(shape=(self.ego_radar.samples, self.ego_radar.chirps), dtype=np.complex128)
        interference_mask = np.ones(shape=(self.ego_radar.samples, self.ego_radar.chirps))
        disturbed = copy.deepcopy(radar_frame)
        for interfering_radar in interfering_radars:
            signal = random.uniform(self.min_power, self.max_power) * self.signal_mutual_interference_cutoff(interfering_radar)
            interference_indicies = self.gen_interference_indicies(interfering_radar)

            for chirp, sample in interference_indicies:
                if chirp >= self.ego_radar.chirps or sample >= self.ego_radar.samples:
                    break
                interference[sample, chirp] += signal[0]
                signal[0] = 0
                num_interference_samples = len(signal)

                for i in range(0, num_interference_samples):
                    if sample + i < self.ego_radar.samples:
                        interference[sample + i, chirp] += signal[i]
                        interference_mask[sample + i, chirp] = 0
                    if sample - i >= 0:
                        interference[sample - i, chirp] += signal[i]
                        interference_mask[sample - i, chirp] = 0

        disturbed += interference

        return radar_frame, interference_mask, disturbed

    def chirp_sawtooth_intersection(self, interfering_radar: InterferingRadar, num_discrete_steps: int = 10000):
        def sawtooth_func(amplitude: float, frequency: float, offset: float):
            def f(t):
                return (
                    amplitude
                    * (2 / 3)
                    * np.arctan(np.tan(2 * np.pi * frequency * t - np.pi / 2))
                    + offset
                )

            return f
 
        time_range = (
            self.ego_radar.time * self.ego_radar.chirps
            if self.ego_radar.time * self.ego_radar.chirps < interfering_radar.time * interfering_radar.chirps
            else interfering_radar.time * interfering_radar.chirps
        )
        t = np.linspace(0, time_range, num_discrete_steps)

        sawtooth1 = sawtooth_func(
            amplitude=self.ego_radar.bw / 2,
            frequency=1 / (self.ego_radar.time * 2),
            offset=self.ego_radar.fc + self.ego_radar.bw / 2,
        )
        sawtooth2 = sawtooth_func(
            amplitude=interfering_radar.bw / 2,
            frequency=1 / (interfering_radar.time * 2),
            offset=interfering_radar.fc + interfering_radar.bw / 2,
        )
        sawtooth1_discrete = np.apply_along_axis(sawtooth1, axis=0, arr=t)
        sawtooth2_discrete = np.apply_along_axis(sawtooth2, axis=0, arr=t)

        sawtooth_diff = sawtooth1_discrete - sawtooth2_discrete

        sawtooth_diff_roots = np.argwhere(
            ((sawtooth_diff[1:] > 0.0) & (sawtooth_diff[: num_discrete_steps - 1] <= 0.0))
            | ((sawtooth_diff[: num_discrete_steps - 1] < 0.0) & (sawtooth_diff[1:] >= 0.0))
        )

        disturbed_frequencies = []
        disturbed_ego_chirp_nums = []
        disturbed_samples = []

        for [root] in sawtooth_diff_roots:
            disturbed_frequencies.append(sawtooth1_discrete[root])
            disturbed_samples.append(round(self.ego_radar.samples * ((sawtooth1_discrete[root] - self.ego_radar.fc) / self.ego_radar.bw)))
            disturbed_ego_chirp_nums.append(
                int(root / (num_discrete_steps / time_range) / self.ego_radar.time)
            )

        return disturbed_ego_chirp_nums, disturbed_samples, disturbed_frequencies

    def gen_interference_indicies(self, interfering_radar):
        disturbed_chirps, disturbed_samples, _ = self.chirp_sawtooth_intersection(interfering_radar)

        return zip(disturbed_chirps, disturbed_samples)

    def gen_interfering_radar(self):
        fc = random.uniform(self.min_fc, self.max_fc)
        bw = random.uniform(self.min_bw, self.max_bw)
        chirps = random.randint(self.min_chirps, self.max_chirps)

        if fc + bw > self.max_fc:
            bw = self.max_fc - fc
        
        return InterferingRadar(
            fc,
            bw,
            chirps,
            random.uniform(self.min_time, self.max_time),
            random.uniform(self.min_power, self.max_power)
        )

    def signal_mutual_interference_cutoff(self, interfering_radar):
        interference_time = (2 * self.ego_radar.lpf_cutoff) / abs(self.ego_radar.slope() - interfering_radar.slope())
        interference_samples = math.ceil(interference_time / self.ego_radar.time_per_sample())

        t = np.linspace(0, interference_time, (interference_samples+1) * 2)
        w_1 = 2*np.pi*self.ego_radar.lpf_cutoff
        w_2 = 2*np.pi*0E06
        s_t = np.sin(w_1 * t + ((w_2-w_1)/interference_time) * np.power(t, 2)/2)

        c = np.fft.fft(s_t)
        c = np.fft.ifft(c[:len(c)//2])
        return c

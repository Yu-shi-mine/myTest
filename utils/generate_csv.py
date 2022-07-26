"""
CSV data Generators
"""

from typing import Tuple
import os, random
from datetime import datetime

import numpy as np
import pandas as pd
import hydra
from omegaconf import DictConfig, OmegaConf


def generate_wave_list(cfg: OmegaConf) -> list[np.ndarray]:
    x = np.arange(0.0, 2.0*np.pi, 2.0*np.pi/cfg.dataset.sequence_num, dtype=np.float32)

    # Generate wave array, num=total_wave_num
    wave_list = []
    for i in range(cfg.dataset.total_wave_num):

        # Sin wave
        wave = np.sin(x) * cfg.dataset.wave_intencity

        # Add random noise if add_noise=true
        if cfg.dataset.add_noise:
            noise = np.random.normal(loc=0., scale=0.05, size=x.shape)
            wave = wave + noise

        # Add constace
        if cfg.dataset.add_shift:
            shift = random.randrange(
                                    cfg.dataset.wave_shift[0],
                                    cfg.dataset.wave_shift[1],
                                    1
                                    ) * cfg.dataset.wave_shift[2]
            wave = wave + shift
        
        # Reshape
        wave = wave[:, np.newaxis]
        
        # Append to list
        wave_list.append(wave)

    return wave_list


def get_min_max(wave_list: list) -> Tuple[float, float]:
    # Convert list to array
    arr = np.array(wave_list)

    # Get min and max
    _min = arr.min()
    _max = arr.max()

    return _min, _max


def normalize_wavedata(wave_list: list, _min: float, _max: float) -> list[np.ndarray]:
    # Normalize wave_list
    norm_wave_list = []
    for wave in wave_list:
        # Min-max scaling
        norm = (wave - _min) / (_max - _min)
        norm_wave_list.append(norm)

    return norm_wave_list


def concat_time(arrays: Tuple, axis: int) -> list[np.ndarray]:
    t = arrays[0]
    wave_list = arrays[1]

    concat_list = []
    for wave in wave_list:
        concat = np.concatenate([t, wave], axis=axis)
        concat_list.append(concat)

    return concat_list


def save_csv(arr_list: list[np.ndarray], save_dt: str, fol_name: str) -> None:
    # Generate save folder
    save_fol = os.path.join(save_dt, fol_name)
    os.makedirs(save_fol, exist_ok=True)

    for i, arr in enumerate(arr_list):
        # Convert list to pd.DataFrame
        np.savetxt(os.path.join(save_fol, '{:02}.csv'.format(i+1)), arr, delimiter=',')

    # NOTE: THIS IS FOR DEBUGGING
    to_concat = []
    for arr in arr_list:
        sub_arr = arr[:, 0]
        to_concat.append(sub_arr)
    df = pd.DataFrame(to_concat)
    df = df.T
    df.to_csv(os.path.join(save_dt, f'{fol_name}_concat.csv'))


@hydra.main(config_name='config', config_path='../config')
def main(cfg: DictConfig):
    # Dataset root folder
    SAVE_ROOT = cfg.dataset.ROOT
    
    # Save folder
    dt = datetime.today().strftime('%Y%m%d_%H%M%S')
    save_dt = os.path.join(SAVE_ROOT, dt)

    # Generate wave data list
    wave_list = generate_wave_list(cfg)

    # Get min and max
    _min, _max = get_min_max(wave_list)

    # Normalize wave data
    norm_wave_list = normalize_wavedata(wave_list, _min, _max)

    # Concat to time label
    t = np.arange(0.0, 1.0, 1.0/cfg.dataset.sequence_num)
    t = t[:, np.newaxis]
    # wave_list = concat_time([t, wave_list], axis=1)
    # norm_wave_list = concat_time([t, norm_wave_list], axis=1)

    # Save as csv
    save_csv(wave_list, save_dt, 'raw_data')
    save_csv(norm_wave_list, save_dt, 'normalized_data')

    # Save min and max
    df = pd.DataFrame([[_min, _max],], index=None, columns=['min', 'max'])
    df.to_csv(os.path.join(save_dt, 'min_max.csv'))

    
if __name__ == '__main__':
    main()

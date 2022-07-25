"""
Dataset Generators
"""

from typing import Tuple

import numpy as np
import hydra
from omegaconf import DictConfig
import torch
from torch.utils.data import Dataset, DataLoader


class WaveDataset(Dataset):
    def __init__(self, x: np.ndarray, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.data, self.label = self.generate_data(x)
    
    def generate_data(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        isFirst = True
        data_list = []
        label_list = []
        for i in range(self.cfg.dataset.total_wave_num):
            # sequence = self.random_sin(x)
            # sequence = self.time_and_sin(x)
            # sequence = self.time_and_cos(x)
            
            sequence = self.time_and_shift_sin(x)
            sub_data, sub_label = self.seq2window(sequence)
            data_list.append(sub_data)
            label_list.append(sub_label)

        data = []
        label = []
        for time in range((self.cfg.dataset.sequence_num-self.cfg.dataset.time_shift) // self.cfg.dataset.time_shift):
            for index in range(len(data_list)):
                data.append(data_list[index][time])
                label.append(label_list[index][time])
        data = np.array(data)
        label = np.array(label)
        return data, label

    def time_and_shift_sin(self, x: np.ndarray) -> np.ndarray:
        sequence : np.ndarray = self.cfg.dataset.wave_intencity * np.sin(x)

        if self.cfg.dataset.add_noise:
            noise = np.random.normal(loc=0., scale=0.05, size=x.shape)
            sequence = sequence + noise
        if sequence.ndim == 1:
            sequence = sequence[:, np.newaxis]

        # Normalization
        sequence = (sequence - sequence.min())/(sequence.max() - sequence.min())
        return sequence
        return

    def random_sin(self, x: np.ndarray) -> np.ndarray:
        sequence : np.ndarray = np.sin(x)
        if self.cfg.dataset.add_noise:
            noise = np.random.normal(loc=0., scale=0.05, size=x.shape)
            sequence = sequence + noise
        if sequence.ndim == 1:
            sequence = sequence[:, np.newaxis]

        # Normalization
        sequence = (sequence - sequence.min())/(sequence.max() - sequence.min())
        return sequence
    
    def time_and_sin(self, x: np.ndarray) -> np.ndarray:
        sequence : np.ndarray = np.sin(x)
        if self.cfg.dataset.add_noise:
            noise = np.random.normal(loc=0., scale=0.05, size=x.shape)
            sequence = sequence + noise

        # Normalization
        sequence = (sequence - sequence.min())/(sequence.max() - sequence.min())

        time = np.arange(0., 1., 1./self.cfg.dataset.sequence_num)
        sequence = np.stack([time, sequence], axis=1)
        return sequence

    def seq2window(self, sequence: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        isFirst = True
        for i in range((sequence.shape[0]-self.cfg.dataset.label_window) // self.cfg.dataset.time_shift):
            window_data = sequence[i:i+self.cfg.dataset.data_window, :]
            if self.cfg.dataset.one2one:
                window_label = sequence[i+self.cfg.dataset.label_window:i+self.cfg.dataset.label_window+1, 1:]
            else:
                window_label = sequence[i+1:i+self.cfg.dataset.label_window+1, 1:]
            window_data = window_data[np.newaxis, :, :]
            window_label = window_label[np.newaxis, :, :]
            if isFirst:
                sub_data = window_data
                sub_label = window_label
                isFirst = False
            else:
                sub_data = np.concatenate([sub_data, window_data], axis=0)
                sub_label = np.concatenate([sub_label, window_label], axis=0)
        return sub_data, sub_label
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index: int) -> Tuple[torch.tensor, torch.tensor]:
        x = torch.tensor(self.data[index], dtype=torch.float32)
        y = torch.tensor(self.label[index], dtype=torch.float32)
        return x, y

@hydra.main(config_name='config', config_path='config')
def main(cfg: DictConfig):
    # Create Dataset
    x = np.arange(0, 2*np.pi, 2*np.pi/cfg.dataset.sequence_num, dtype=np.float32)
    dataset = WaveDataset(x, cfg)

    # Create DataLoader
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=cfg.dataset.total_wave_num,
        shuffle=cfg.dataloader.shuffle,
        num_workers=cfg.dataloader.num_worklers
    )

    iterator = iter(dataloader)
    data, label = next(iterator)

    print(data.shape)
    print(label.shape)
    print(data)
    print(label)


# Test
if __name__ =='__main__':
    main()

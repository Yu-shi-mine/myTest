"""
Dataset Generators
"""

from typing import Tuple
import os, glob

import numpy as np
import hydra
from omegaconf import DictConfig
import torch
from torch.utils.data import Dataset, DataLoader


class CSVDataset(Dataset):
    def __init__(self, csv_list: list[str], cfg: DictConfig) -> None:
        self.cfg = cfg
        self.data, self.label = self.generate_data(csv_list)
    
    def generate_data(self, csv_list: list[str]) -> Tuple[np.ndarray, np.ndarray]:
        data_list = []
        label_list = []
        for csv in csv_list:
            # Load csv data as np.ndarray
            array = np.loadtxt(csv, dtype=np.float32, delimiter=',')
            if array.ndim == 1:
                array = array[:, np.newaxis]

            # Convert to windowed data and label
            sub_data, sub_label = self.seq2window(array)

            # Append to list
            data_list.append(sub_data)
            label_list.append(sub_label)

        # Sort
        data = []
        label = []
        for j in range((self.cfg.dataset.sequence_num-self.cfg.dataset.label_window) // self.cfg.dataset.time_shift):
            for i in range(len(data_list)):
                data.append(data_list[i][j])
                label.append(label_list[i][j])
        
        data = np.array(data)
        label = np.array(label)

        return data, label

    def seq2window(self, array: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        isFirst = True
        for i in range((array.shape[0]-self.cfg.dataset.label_window) // self.cfg.dataset.time_shift):
            window_data = array[i:i+self.cfg.dataset.data_window, :]
            window_label = array[i+self.cfg.dataset.label_window:i+self.cfg.dataset.label_window+1, self.cfg.dataset.label_window-1:]

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
    dataset_root = os.path.join(cfg.dataset.ROOT, cfg.dataset.fol_name)
    csv_list = glob.glob(os.path.join(dataset_root, 'train/*.csv'))

    dataset = CSVDataset(csv_list, cfg)

    # Create DataLoader
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=cfg.dataset.train_wave_num,
        shuffle=cfg.dataset.shuffle,
        num_workers=0
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

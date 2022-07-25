"""
Test trained LSTM model
"""

from typing import Tuple
from datetime import datetime
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import hydra
from omegaconf import DictConfig
import mlflow
from mlflow import log_metric, log_param, log_artifacts
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader

from dataset import WaveDataset
from model.model import One2OneLSTM, One2OneStatefulLSTM, One2ManyStatefulLSTM


@hydra.main(config_name='config', config_path='config')
def main(cfg: DictConfig):
    # Output folder
    dt = datetime.today().strftime('%Y%m%d_%H%M%S')
    result_dir = os.path.join(cfg.infer.weight_root, cfg.infer.weigth_folder, 'result')
    os.makedirs(result_dir, exist_ok=True)

    # Define model
    if cfg.model.name == 'One2One':
        model = One2OneLSTM(cfg)
    elif cfg.model.name == 'One2OneStateful':
        model = One2OneStatefulLSTM(cfg)
    elif cfg.model.name == 'One2ManyStateful':
        model = One2ManyStatefulLSTM(cfg)
    
    weights = ['100', '300', '500']
    for weight in weights:
        # Load trained weights
        weights = torch.load(os.path.join(cfg.infer.weight_root, cfg.infer.weigth_folder, 'weights_'+weight+'.pth'), map_location={'cuda:0':'cpu'})
        model.load_state_dict(weights)

        # Create test DataLoader
        dataloader = gen_dataloader(cfg)

        # Infer
        if not cfg.infer.recursive:
            result = infer(cfg, model, dataloader)
        else:
            result = recursive_infer(cfg, model, dataloader)

        # Inverse Normalization
        result = result * 2 - 1
        
        # Save results
        if cfg.model.input_size == 1:
            col_name = ['sin_pred', 'sin_truth']
        elif cfg.model.input_size == 2:
            col_name = ['sin_pred', 'sin_truth']
        df = pd.DataFrame(result, index=None, columns=col_name)
        df.sort_index(axis=1, inplace=True)

        if cfg.infer.recursive:
            name = 'recursive'
        else:
            name = 'non_recursive'
        df.to_csv(os.path.join(result_dir, f'{weight}epoch_{dt}_{name}.csv'))


def infer(cfg: DictConfig, model: nn.Module, dataloader: DataLoader) -> np.ndarray:
    with torch.no_grad():
        isFirst = True
        for x, y in dataloader:
            # Initialize hidden and cell state for the first step
            if isFirst:
                hn = torch.zeros(size=[1, 1, cfg.model.hidden_size])
                cn = torch.zeros(size=[1, 1, cfg.model.hidden_size])
            
            # infer
            if cfg.model.stateful:
                outputs, hn, cn = model(x, hn, cn)
            else:
                outputs = model(x)

            # Save results
            pred = outputs[:, 0, :].numpy()
            truth = y[:, 0, :].numpy()
            returns = np.concatenate([pred, truth], axis=1)

            if isFirst:
                result = returns
                isFirst = False
            else:
                result = np.concatenate([result, returns], axis=0)
    return result


def recursive_infer(cfg: DictConfig, model: nn.Module, dataloader: DataLoader) -> np.ndarray:
    with torch.no_grad():
        isFirst = True
        for x, y in dataloader:
            # Initialize hidden and cell state for the first step
            if isFirst:
                hn = torch.zeros(size=[1, 1, cfg.model.hidden_size])
                cn = torch.zeros(size=[1, 1, cfg.model.hidden_size])

            # Prepare the input
            if isFirst:
                _input = x
            else:
                next_t = x[:, -1, 0:1].unsqueeze(1)
                next_feature = torch.cat([next_t, next_x], axis=2)
                _input = torch.cat([previos_input, next_feature], axis=1)
            # print('--------------------------------------')
            # print('x =', x)
            # print('input =', _input)
            # if not isFirst:
            #     print('output =', output)
            
            # infer
            if cfg.model.stateful:
                output, hn, cn = model(_input, hn, cn)
            else:
                output = model(_input)

            # Save results
            pred = output[:, 0, :].numpy()
            truth = y[:, 0, :].numpy()
            returns = np.concatenate([pred, truth], axis=1)

            if isFirst:
                result = returns
                isFirst = False
            else:
                result = np.concatenate([result, returns], axis=0)

            # Prepare for next step
            previos_input = _input[:, 1:, :]
            next_x = output[:, -1, :].unsqueeze(1)
    return result


def gen_dataloader(cfg: DictConfig) -> DataLoader:
    # Create Dataset
    x = np.arange(0, 2*np.pi, 2*np.pi/cfg.dataset.sequence_num, dtype=np.float32)
    test_dataset = WaveDataset(x, cfg)

    # Create DataLoader
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=cfg.dataloader.shuffle,
        num_workers=cfg.dataloader.num_worklers
    )
    return test_dataloader


# Run
if __name__ =='__main__':
    main()
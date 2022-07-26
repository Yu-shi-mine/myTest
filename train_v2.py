"""
Train LSTM
"""

from typing import Tuple
import warnings
from datetime import datetime
import os, time, glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import hydra
from omegaconf import DictConfig
import mlflow
from mlflow import log_param, log_metric, log_artifacts
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader

from dataset import CSVDataset
from model.model import One2OneLSTM, One2OneStatefulLSTM, One2ManyStatefulLSTM


@hydra.main(config_name='config', config_path='config')
def main(cfg: DictConfig):
    hydra_path = os.getcwd()
    # Fix seed
    torch.manual_seed(0000)
    torch.cuda.manual_seed(0000)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True

    # Output folder
    dt = datetime.today().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(cfg.train.log_root, dt+'_v2')
    os.makedirs(log_dir, exist_ok=True)

    # Create mlflow experiment
    mlflow.set_tracking_uri(cfg.mlflow.uri)
    experiment = mlflow.get_experiment_by_name(cfg.mlflow.run_name)
    if experiment is None:
        my_id = mlflow.create_experiment(
            name=cfg.mlflow.run_name,
            artifact_location=cfg.mlflow.work_dir
        )
    else:
        my_id = experiment.experiment_id

    # Start mlflow tracking
    with mlflow.start_run(experiment_id=my_id):
        # Track mlflow params
        log_param('sequence_num', cfg.dataset.sequence_num)
        log_param('data_window', cfg.dataset.data_window)
        log_param('label_window', cfg.dataset.label_window)
        log_param('total_wave_num', cfg.dataset.total_wave_num)
        log_param('num_epochs', cfg.train.num_epochs)
        log_param('input_size', cfg.model.input_size)
        log_param('hidden_size', cfg.model.hidden_size)
        log_param('output_size', cfg.model.output_size)

        # Define model
        if cfg.model.name == 'One2One':
            model = One2OneLSTM(cfg)
        elif cfg.model.name == 'One2OneStateful':
            model = One2OneStatefulLSTM(cfg)
        elif cfg.model.name == 'One2ManyStateful':
            model = One2ManyStatefulLSTM(cfg)

        # Define optimizer
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        # Define LR scheduler
        scheduler = ExponentialLR(optimizer, gamma=0.95)

        # Device configuration
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        # Create Dataset
        dataloaders_dict = gen_dataloaders(cfg)
        
        # Iteration counter
        train_iteration = 1
        val_iteration = 1
        epoch_train_loss = 0.0
        epoch_val_loss = 0.0
        logs = []
        epoch_duration = 0.0

        # Epoch loop
        # Record time first epoch start
        t_epoch_start = time.time()
        for epoch in range(cfg.train.num_epochs):
            print('--------------------------------------')

            # Train and val loop
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()
                
                # Mini batch Loop
                for i, (x, y) in enumerate(tqdm(dataloaders_dict[phase], desc=f'{epoch+1}/{cfg.train.num_epochs}')):
                    # Initialize optimizer
                    optimizer.zero_grad()

                    # Initialize hidden and cell state
                    if i == 0:
                        hn = torch.zeros(size=[1, cfg.dataset.total_wave_num, cfg.model.hidden_size])
                        cn = torch.zeros(size=[1, cfg.dataset.total_wave_num, cfg.model.hidden_size])
                        
                    # Prepare tha input tensor
                    if i == 0:
                        _input = x
                    else:
                        input_time: torch.Tensor = x[:, :, 0:1]
                        input_time = input_time.to(device)
                        _input = torch.cat([input_time, next_head], axis=2)

                    # Send to device
                    _input: torch.Tensor = _input.to(device)
                    y: torch.Tensor = y.to(device)
                    hn: torch.Tensor = hn.to(device)
                    cn: torch.Tensor = cn.to(device)

                    # Forward
                    with torch.set_grad_enabled(phase == 'train'):

                        output, hn, cn = model(_input, hn, cn)
                        # output= model(x)

                        # Caluculate loss
                        loss = F.mse_loss(output, y)

                        # Backpropagation when phase == train
                        if phase == 'train':
                            loss.backward()

                            nn.utils.clip_grad_value_(
                                model.parameters(), clip_value=2.0
                            )

                            # Step optimizer
                            optimizer.step()

                            epoch_train_loss += loss.item()
                            train_iteration += 1
                        else:
                            epoch_val_loss += loss.item()
                            val_iteration += 1
                    
                    # Prepare for next step
                    next_head = output.detach()[:, :, 0:]

            # step LR scheduler
            scheduler.step()

            # Caluculate average loss
            train_loss = epoch_train_loss/train_iteration
            val_loss = epoch_val_loss/val_iteration

            # Save mlflow metrics
            log_metric('train_loss', train_loss, step=1)
            log_metric('val_loss', val_loss, step=1)

            # Display
            t_epoch_finish = time.time()
            epoch_duration = t_epoch_finish - t_epoch_start
            print('epoch: {} || loss: {:e} || val_loss: {:e}'.format(epoch+1, train_loss, val_loss))
            print('timer: {:.4f} sec.'.format(epoch_duration))
            t_epoch_start = time.time()
            train_iteration = 1
            val_iteration = 1

            # Save logs
            log_epoch = {
                'epoch': epoch+1,
                'loss': train_loss,
                'val_loss': val_loss
            }
            logs.append(log_epoch)
            df = pd.DataFrame(logs)
            df.to_csv(os.path.join(log_dir, 'log.csv'), index=0)

            epoch_train_loss = 0.0
            epoch_val_loss = 0.0

            # Save checkpoints every 10 epochs
            if ((epoch+1) % 10 == 0):
                torch.save(model.state_dict(), os.path.join(log_dir, './weights_' + str(epoch+1) + '.pth'))


def gen_dataloaders(cfg: DictConfig) -> dict[str: DataLoader, str: DataLoader]:
    # Get csv list
    dataset_root = os.path.join(cfg.dataset.ROOT, cfg.dataset.fol_name)
    train_csv_list = glob.glob(os.path.join(dataset_root, 'train/*.csv'))
    val_csv_list = glob.glob(os.path.join(dataset_root, 'val/*.csv'))

    # Create Dataset
    train_dataset = CSVDataset(train_csv_list, cfg)
    val_dataset = CSVDataset(val_csv_list, cfg)

    # Create DataLoader
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg.dataset.train_wave_num,
        shuffle=cfg.dataset.shuffle,
        num_workers=0
    )
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=cfg.dataset.val_wave_num,
        shuffle=cfg.dataset.shuffle,
        num_workers=0
    )
    dataloaders_dict = {'train': train_dataloader, 'val': val_dataloader}
    return dataloaders_dict


# Run
if __name__ =='__main__':
    warnings.simplefilter('ignore')
    main()

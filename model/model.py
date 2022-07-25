"""
LSTM model
"""

from typing import Tuple

import hydra
from omegaconf import DictConfig
import torch
import torch.nn as nn
from torchinfo import summary


class One2OneLSTM(nn.Module):
    def __init__(self, cfg: DictConfig) -> None:
        super(One2OneLSTM, self).__init__()
        self.lstm_layer = nn.LSTM(cfg.model.input_size, cfg.model.hidden_size, batch_first=True, bidirectional=cfg.model.bidirectional)
        if cfg.model.bidirectional:
            self.output_layer = nn.Linear(in_features=cfg.model.hidden_size*2, out_features=cfg.model.output_size)
        else:
            self.output_layer = nn.Linear(in_features=cfg.model.hidden_size, out_features=cfg.model.output_size)

    def forward(self, x) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
        x, (hn, cn) = self.lstm_layer(x)
        x = self.output_layer(x)
        return x


class One2OneStatefulLSTM(nn.Module):
    def __init__(self, cfg: DictConfig) -> None:
        super(One2OneStatefulLSTM, self).__init__()
        self.input_layer = nn.Linear(in_features=cfg.model.input_size, out_features=cfg.model.lstm_input)
        self.lstm_layer = nn.LSTM(cfg.model.lstm_input, cfg.model.hidden_size, batch_first=True)
        self.output_layer = nn.Linear(in_features=cfg.model.hidden_size, out_features=cfg.model.output_size)

    def forward(self, x, hn, cn) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
        x = self.input_layer(x)
        x, (hn, cn) = self.lstm_layer(x, (hn.detach(), cn.detach()))
        x = self.output_layer(x)
        return x, hn, cn


class One2ManyStatefulLSTM(nn.Module):
    def __init__(self, cfg: DictConfig) -> None:
        super(One2ManyStatefulLSTM, self).__init__()
        # Stacked LSTM
        self.lstm_1 = nn.LSTM(cfg.model.input_size, cfg.model.hidden_size, batch_first=True)
        self.lstm_2 = nn.LSTM(cfg.model.hidden_size, cfg.model.hidden_size, batch_first=True)
        self.lstm_3 = nn.LSTM(cfg.model.hidden_size, cfg.model.hidden_size, batch_first=True)
        self.lstm_4 = nn.LSTM(cfg.model.hidden_size, cfg.model.hidden_size, batch_first=True)
        self.lstm_5 = nn.LSTM(cfg.model.hidden_size, cfg.model.hidden_size, batch_first=True)

        # Output Layers
        self.output_1 = nn.Linear(in_features=cfg.model.hidden_size, out_features=cfg.model.output_size)
        self.output_2 = nn.Linear(in_features=cfg.model.hidden_size, out_features=cfg.model.output_size)
        self.output_3 = nn.Linear(in_features=cfg.model.hidden_size, out_features=cfg.model.output_size)
        self.output_4 = nn.Linear(in_features=cfg.model.hidden_size, out_features=cfg.model.output_size)
        self.output_5 = nn.Linear(in_features=cfg.model.hidden_size, out_features=cfg.model.output_size)

    def forward(self, x, hn, cn) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
        x, (hn_1, cn_1) = self.lstm_1(x, (hn.detach(), cn.detach()))
        out_1 = self.output_1(x)

        x, (hn_2, cn_2) = self.lstm_2(x, (hn_1, cn_1))
        out_2 = self.output_2(x)

        x, (hn_3, cn_3) = self.lstm_3(x, (hn_2, cn_2))
        out_3 = self.output_3(x)

        x, (hn_4, cn_4) = self.lstm_4(x, (hn_3, cn_3))
        out_4 = self.output_4(x)

        x, (hn_5, cn_5) = self.lstm_5(x, (hn_4, cn_4))
        out_5 = self.output_5(x)

        outputs = torch.cat([out_1, out_2, out_3, out_4, out_5], dim=1)
        return outputs, hn_1, cn_1


@hydra.main(config_name='config', config_path='../config')
def main(cfg: DictConfig):
    model = One2OneLSTM(cfg)

    x = torch.rand(size=[cfg.dataloader.batch_size, cfg.dataset.data_window, cfg.model.input_size])
    hn = torch.zeros(size=[1, cfg.dataloader.batch_size, cfg.model.hidden_size])
    cn = torch.zeros(size=[1, cfg.dataloader.batch_size, cfg.model.hidden_size])

    output, hn, cn = model(x, hn, cn)

    print(output.shape)
    print(hn.shape)
    print(cn.shape)

    print(summary(model))


# Test
if __name__ == '__main__':
    main()

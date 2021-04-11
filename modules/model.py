import torch
import torch.nn as nn


class Q_Network(torch.nn.Module):
    def __init__(self, input_size, n_feachs):
        super(Q_Network, self).__init__()
        self.input_size = input_size
        self.n_feachs = n_feachs
        padding = 1
        dropout = 0.3
        self.backbone = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=1, out_channels=2, kernel_size=2, dilation=1, bias=True, padding=padding),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(2),
            torch.nn.Dropout(dropout),
            torch.nn.Conv1d(in_channels=2, out_channels=4, kernel_size=2, dilation=2, bias=True, padding=padding),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(4),
            torch.nn.Dropout(dropout),
            torch.nn.Conv1d(in_channels=4, out_channels=8, kernel_size=2, dilation=4, bias=True, padding=padding),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(8),
            torch.nn.Dropout(dropout),
            torch.nn.Conv1d(in_channels=8, out_channels=16, kernel_size=2, dilation=8, bias=True, padding=padding),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(16),
            torch.nn.Dropout(dropout),
            torch.nn.Conv1d(in_channels=16, out_channels=32, kernel_size=2, dilation=16, bias=True, padding=padding),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(32),
            torch.nn.Dropout(dropout),
            torch.nn.Conv1d(in_channels=32, out_channels=64, kernel_size=2, dilation=32, bias=True, padding=padding),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(64),
            torch.nn.Dropout(dropout),
            # torch.nn.Conv1d(in_channels=64, out_channels=128, kernel_size=2, dilation=64, bias=True, padding=padding),
            # torch.nn.ReLU(),
            # torch.nn.BatchNorm1d(128),
            # torch.nn.Dropout(dropout),
        )

        self.affine_layer = torch.nn.Sequential(
            torch.nn.Linear(64 * 3, 2048),
            torch.nn.SELU(),
            torch.nn.AlphaDropout(0.2),
            torch.nn.Linear(2048, 1024),
            torch.nn.SELU(),
            torch.nn.AlphaDropout(0.2),
            torch.nn.Linear(1024, 512),
            torch.nn.SELU(),
            torch.nn.AlphaDropout(0.2),
            torch.nn.Linear(512, 3),
            torch.nn.SELU(),
            torch.nn.AlphaDropout(0.2),
        )

    def reset(self):
        self.zero_grad()

    def forward(self, x):
        print("simple network")
        x = self.backbone(x.view(-1, 1, self.input_size*self.n_feachs))
        print(x.shape)
        x = self.affine_layer(x.view(-1, 64 * 54))

        return x


class Dueling_Q_Network(nn.Module):
    def __init__(self, input_size, n_feachs):
        super(Dueling_Q_Network, self).__init__()
        self.input_size = input_size
        self.n_feachs = n_feachs
        padding = 0
        dropout = 0.2
        self.backbone = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=1, out_channels=2, kernel_size=2, dilation=1, bias=True, padding=padding),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(2),
            torch.nn.Dropout(dropout),
            torch.nn.Conv1d(in_channels=2, out_channels=4, kernel_size=2, dilation=2, bias=True, padding=padding),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(4),
            torch.nn.Dropout(dropout),
            torch.nn.Conv1d(in_channels=4, out_channels=8, kernel_size=2, dilation=4, bias=True, padding=padding),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(8),
            torch.nn.Dropout(dropout),
            torch.nn.Conv1d(in_channels=8, out_channels=16, kernel_size=2, dilation=8, bias=True, padding=padding),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(16),
            torch.nn.Dropout(dropout),
            torch.nn.Conv1d(in_channels=16, out_channels=32, kernel_size=2, dilation=16, bias=True, padding=padding),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(32),
            torch.nn.Dropout(dropout),
            torch.nn.Conv1d(in_channels=32, out_channels=64, kernel_size=2, dilation=32, bias=True, padding=padding),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(64),
            torch.nn.Dropout(dropout),
            # torch.nn.Conv1d(in_channels=64, out_channels=128, kernel_size=2, dilation=64, bias=True, padding=padding),
            # torch.nn.ReLU(),
            # torch.nn.BatchNorm1d(128),
            # torch.nn.Dropout(dropout),
        )

        self.state_value = torch.nn.Sequential(
            torch.nn.Linear(self.input_size * 129, 2048),
            torch.nn.SELU(),
            torch.nn.AlphaDropout(dropout),
            torch.nn.Linear(2048, 1024),
            torch.nn.SELU(),
            torch.nn.AlphaDropout(dropout),
            torch.nn.Linear(1024, 512),
            torch.nn.SELU(),
            torch.nn.AlphaDropout(dropout),
            torch.nn.Linear(512, 1),
            torch.nn.SELU(),
            torch.nn.AlphaDropout(dropout),
        )

        self.advantage_value = torch.nn.Sequential(
            torch.nn.Linear(self.input_size * 129, 2048),
            torch.nn.SELU(),
            torch.nn.AlphaDropout(dropout),
            torch.nn.Linear(2048, 1024),
            torch.nn.SELU(),
            torch.nn.AlphaDropout(dropout),
            torch.nn.Linear(1024, 512),
            torch.nn.SELU(),
            torch.nn.AlphaDropout(dropout),
            torch.nn.Linear(512, 3),
            torch.nn.SELU(),
            torch.nn.AlphaDropout(dropout),
        )

    def reset(self):
        self.zero_grad()

    def forward(self, x):

        x = self.backbone(x.reshape(-1, 1, self.input_size*self.n_feachs))

        state_value = self.state_value(x.view(-1, 129*self.input_size))
        advantage_value = self.advantage_value(x.view(-1, 129*self.input_size))
        advantage_mean = torch.Tensor.mean(advantage_value, dim=1, keepdim=True)
        q_value = state_value.expand([-1, 3]) + (advantage_value - advantage_mean.expand([-1, 3]))

        return q_value

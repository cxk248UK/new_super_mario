import copy
import math

import torch
from torch import nn, Tensor
from custom_common_dict import USE_DEVICE


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 4):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size,seq_len embedding_dim]``
        """
        x = x + self.pe[0, :x.size(0)]
        return self.dropout(x)


class MiniCnnModel(nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()
        c, h, w = input_dim

        self.online = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

        self.target = copy.deepcopy(self.online)

        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, input_frame, model='online'):
        input_frame = input_frame.to(torch.float)
        if model == 'online':
            return self.online(input_frame)
        elif model == 'target':
            return self.target(input_frame)


class MiniSimplifyCnnModel(nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()
        c, h, w = input_dim

        self.online = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2592, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

        self.target = copy.deepcopy(self.online)

        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, input_frame, model='online'):
        input_frame = input_frame.to(torch.float)
        if model == 'online':
            return self.online(input_frame)
        elif model == 'target':
            return self.target(input_frame)


class MiniComplexCnnLayerModel(nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()
        c, h, w = input_dim

        self.online = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

        self.target = copy.deepcopy(self.online)

        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, input_frame, model='online'):
        input_frame = input_frame.to(torch.float)
        if model == 'online':
            return self.online(input_frame)
        elif model == 'target':
            return self.target(input_frame)


class MiniTransformerCnnModel(nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()
        c, h, w = input_dim

        self.online_cnn = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=4 * 6 * 6, kernel_size=6, stride=6),
            nn.Flatten(-2, -1),
            nn.LayerNorm(196)
        )

        self.online_transformer = nn.Sequential(
            PositionalEncoding(196, max_len=144),
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=196, dim_feedforward=512, nhead=2, batch_first=True), 1),
            nn.Flatten(-2, -1),
            nn.Linear(in_features=144 * 196, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=output_dim)
        )

        self.target_cnn = copy.deepcopy(self.online_cnn)

        self.target_transformer = copy.deepcopy(self.online_transformer)

        for p in self.target_cnn.parameters():
            p.requires_grad = False

        for p in self.target_transformer.parameters():
            p.requires_grad = False

    def forward(self, input_frame, model='online'):
        input_frame = input_frame.to(torch.float)
        if model == 'online':
            x = self.online_cnn(input_frame)
            return self.online_transformer(x)
        elif model == 'target':
            x = self.target_cnn(input_frame)
            return self.target_transformer(x)


class MiniSimplifyTransformerCnnModel(nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()
        c, h, w = input_dim

        self.online_cnn = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=3, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Flatten(-3, -1),
            nn.Linear(1200, 512)
        )

        self.online_transformer = nn.Sequential(
            PositionalEncoding(512),
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=512, dim_feedforward=512, nhead=1, batch_first=True), 1),
            nn.Flatten(-2, -1),
            nn.Linear(in_features=2048, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=output_dim)
        )

        self.target_cnn = copy.deepcopy(self.online_cnn)

        self.target_transformer = copy.deepcopy(self.online_transformer)

        for p in self.target_cnn.parameters():
            p.requires_grad = False

        for p in self.target_transformer.parameters():
            p.requires_grad = False

    def forward(self, input_frame, model='online'):
        input_frame = input_frame.to(torch.float)
        input_frame = input_frame.unsqueeze(2)
        transformer_input = torch.zeros((1, 4, 512)).to(device=USE_DEVICE)
        if model == 'online':
            for seq_pic in input_frame:
                transformer_input = torch.cat((transformer_input, self.online_cnn(seq_pic).unsqueeze(0)), 0)
            transformer_input = transformer_input[1:]
            return self.online_transformer(transformer_input)
        elif model == 'target':
            for seq_pic in input_frame:
                transformer_input = torch.cat((transformer_input, self.target_cnn(seq_pic).unsqueeze(0)), 0)
            transformer_input = transformer_input[1:]
            return self.target_transformer(transformer_input)


class MiniComplexityTransformerCnnModel(nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()
        c, h, w = input_dim

        self.online_cnn = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=3, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Flatten(-3, -1),
            nn.Linear(1200, 512)
        )

        self.online_transformer = nn.Sequential(
            PositionalEncoding(512),
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=512, dim_feedforward=512, nhead=4, batch_first=True), 1),
            nn.Flatten(-2, -1),
            nn.Linear(in_features=2048, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=output_dim)
        )

        self.target_cnn = copy.deepcopy(self.online_cnn)

        self.target_transformer = copy.deepcopy(self.online_transformer)

        for p in self.target_cnn.parameters():
            p.requires_grad = False

        for p in self.target_transformer.parameters():
            p.requires_grad = False

    def forward(self, input_frame, model='online'):
        input_frame = input_frame.to(torch.float)
        input_frame = input_frame.unsqueeze(2)
        transformer_input = torch.zeros((1, 4, 512)).to(device=USE_DEVICE)
        if model == 'online':
            for seq_pic in input_frame:
                transformer_input = torch.cat((transformer_input, self.online_cnn(seq_pic).unsqueeze(0)), 0)
            transformer_input = transformer_input[1:]
            return self.online_transformer(transformer_input)
        elif model == 'target':
            for seq_pic in input_frame:
                transformer_input = torch.cat((transformer_input, self.target_cnn(seq_pic).unsqueeze(0)), 0)
            transformer_input = transformer_input[1:]
            return self.target_transformer(transformer_input)

import torch
from torch import nn
from CNN import PositionalEncoding, MiniTransformerCnnModel

test_model = MiniTransformerCnnModel((4, 84, 84), 3)

src = torch.randn(32, 4, 84, 84)

test_output = test_model(src)
#
# src = src.unsqueeze(2)
#
# for seq_pic in src:
#     print(seq_pic.shape)
#
# transformer_input = torch.zeros((1, 4, 512))
#
# for seq_pic in src:
#     transformer_input = torch.cat((transformer_input, test_model(seq_pic).unsqueeze(0)), 0)
#
# transformer_input = transformer_input[1:]
#
# position_encoding = PositionalEncoding(d_model=512)
#
# transformer_input = position_encoding(transformer_input)
#
# transformer_encoding_layer = nn.TransformerEncoderLayer(d_model=512, dim_feedforward=1024, nhead=4, batch_first=True)
#
# transformer_encoder = nn.TransformerEncoder(transformer_encoding_layer, 2)
#
# transformer_output = transformer_encoder(transformer_input)

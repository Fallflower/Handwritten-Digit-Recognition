from torch import nn
import torch
import numpy as np


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 7**2, 128),
            # nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(128, 10),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # print(x.shape)
        x = self.fc(x.view(x.shape[0], -1))
        return x


class AttentionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers):
        super(AttentionModel, self).__init__()

        # 定义自注意力层
        self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads)

        # 定义前馈全连接层
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.ReLU(),
            nn.Linear(4 * hidden_dim, hidden_dim)
        )

        # 定义残差连接层和层归一化层
        self.residual_norm1 = nn.LayerNorm(hidden_dim)
        self.residual_norm2 = nn.LayerNorm(hidden_dim)

        # 定义位置编码
        self.position_encoding = PositionEncoding(hidden_dim)

        # 定义编码层
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(hidden_dim, num_heads) for _ in range(num_layers)
        ])

    def forward(self, inputs):
        batch_size, seq_len, _ = inputs.size()

        # 添加位置编码
        inputs += self.position_encoding(seq_len)

        # 自注意力层
        self_attention_output, _ = self.self_attention(inputs, inputs, inputs)
        self_attention_output = self.residual_norm1(inputs + self_attention_output)

        # 前馈全连接层
        feed_forward_output = self.feed_forward(self_attention_output)
        feed_forward_output = self.residual_norm2(self_attention_output + feed_forward_output)

        return feed_forward_output


class PositionEncoding(nn.Module):
    def __init__(self, hidden_dim, max_seq_len=100):
        super(PositionEncoding, self).__init__()

        position_encoding = torch.zeros(max_seq_len, hidden_dim)
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, hidden_dim, 2).float() * (-np.log(10000.0) / hidden_dim))

        position_encoding[:, 0::2] = torch.sin(position * div_term)
        position_encoding[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('position_encoding', position_encoding)

    def forward(self, seq_len):
        return self.position_encoding[:seq_len, :]


class EncoderLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super(EncoderLayer, self).__init__()

        self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.ReLU(),
            nn.Linear(4 * hidden_dim, hidden_dim)
        )
        self.residual_norm1 = nn.LayerNorm(hidden_dim)
        self.residual_norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, inputs):
        self_attention_output, _ = self.self_attention(inputs, inputs, inputs)
        self_attention_output = self.residual_norm1(inputs + self_attention_output)

        feed_forward_output = self.feed_forward(self_attention_output)
        feed_forward_output = self.residual_norm2(self_attention_output + feed_forward_output)

        return feed_forward_output

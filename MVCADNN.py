#!/Users/11834/.conda/envs/Pytorch_GPU/python.exe
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File    ：RNASolventAccessibility -> MVCAC
@IDE    ：PyCharm
@Date   ：2021/5/9 15:53
=================================================='''

import torch.nn as nn
import math
import torch


class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(channel)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid())

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc(y)
        return x * y.expand_as(x)


class BasicBlock(nn.Module):
    def __init__(self, input, output):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv1d(input, output, kernel_size=5, padding=4, dilation=2),
                                   nn.BatchNorm1d(output),
                                   nn.ELU(inplace=True),
                                   nn.Dropout(0.4),

                                   nn.Conv1d(input, output, kernel_size=7, padding=6, dilation=2),
                                   nn.BatchNorm1d(output),
                                   nn.ELU(inplace=True),
                                   nn.Dropout(0.4))

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out += residual
        return out


class LSTMMergeSE(nn.Module):

    def __init__(self, num_layers=2, input_dim_0=100, hidden_dim_0=128, input_dim_1=100, hidden_dim_1=128,
                 in_channels_2=64, out_channels_2=64, input_dim_3=25, hidden_dim_3=16):
        super(LSTMMergeSE, self).__init__()
        self.num_layers = num_layers
        self.input_dim_0 = input_dim_0
        self.hidden_dim_0 = hidden_dim_0
        self.input_dim_1 = input_dim_1
        self.hidden_dim_1 = hidden_dim_1
        self.in_channels_2 = in_channels_2
        self.out_channels_2 = out_channels_2
        self.input_dim_3 = input_dim_3
        self.hidden_dim_3 = hidden_dim_3
        # self.ResNet = BasicBlock(self.in_channels_2, self.out_channels_2)

        self.lstm00 = nn.LSTM(input_size=self.input_dim_0, hidden_size=self.hidden_dim_0,
                              num_layers=self.num_layers,
                              dropout=0.5, batch_first=True, bidirectional=True)
        self.lstm01 = nn.LSTM(input_size=self.hidden_dim_0 * 2, hidden_size=self.hidden_dim_0 * 2,
                              num_layers=self.num_layers,
                              dropout=0.5, batch_first=True, bidirectional=True)

        self.SELayer00 = SELayer(512)
        self.fc00 = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            GELU(),
            nn.Dropout(0.5),
        )
        self.SELayer01 = SELayer(256)
        self.fc01 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            GELU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )

        self.lstm10 = nn.LSTM(input_size=self.input_dim_1, hidden_size=self.hidden_dim_1,
                              num_layers=self.num_layers,
                              dropout=0.5, batch_first=True, bidirectional=True)
        self.lstm11 = nn.LSTM(input_size=self.hidden_dim_1 * 2, hidden_size=self.hidden_dim_1 * 2,
                              num_layers=self.num_layers,
                              dropout=0.5, batch_first=True, bidirectional=True)
        self.SELayer10 = SELayer(512)
        self.fc10 = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            GELU(),
            nn.Dropout(0.5),
        )
        self.SELayer11 = SELayer(256)
        self.fc11 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            GELU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )
        self.conv1 = nn.Sequential(nn.Conv1d(in_channels=9, out_channels=64, kernel_size=3, padding=1, bias=False),
                                   nn.BatchNorm1d(64),
                                   nn.ELU(inplace=True))
        self.ResNet3 = nn.Sequential(BasicBlock(self.in_channels_2, self.out_channels_2),
                                     BasicBlock(self.in_channels_2, self.out_channels_2),
                                     BasicBlock(self.in_channels_2, self.out_channels_2),
                                     nn.BatchNorm1d(64),
                                     nn.ELU(inplace=True),
                                     nn.Dropout(0.4))
        self.fc20 = nn.Sequential(nn.Linear(64, 32),
                                  nn.BatchNorm1d(32),
                                  nn.ELU(inplace=True),
                                  nn.Dropout(0.5),
                                  nn.Linear(32, 1))

        self.lstm30 = nn.LSTM(input_size=self.input_dim_3, hidden_size=self.hidden_dim_3,
                              num_layers=self.num_layers,
                              dropout=0.5, batch_first=True, bidirectional=True)
        self.lstm31 = nn.LSTM(input_size=self.hidden_dim_3 * 2, hidden_size=self.hidden_dim_3 * 2,
                              num_layers=self.num_layers,
                              dropout=0.5, batch_first=True, bidirectional=True)
        self.SELayer30 = SELayer(64)
        self.fc30 = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            GELU(),
            nn.Dropout(0.5),
        )
        self.SELayer31 = SELayer(32)
        self.fc31 = nn.Sequential(
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            GELU(),
            nn.Dropout(0.5),
            nn.Linear(16, 1)
        )

        self.SELayer4 = SELayer(608)
        self.fc4 = nn.Sequential(
            nn.Linear(608, 256),
            nn.BatchNorm1d(256),
            GELU(),
            nn.Dropout(0.5))
        self.SELayer5 = SELayer(256)
        self.fc5 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            GELU(),
            nn.Dropout(0.5))

        self.SELayer6 = SELayer(128)
        self.fc6 = nn.Sequential(nn.Linear(128, 1))

        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.fill_(0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0, 0.01)
                if m.bias is not None:
                    m.bias.data.fill_(0)

    def forward(self, x0, x1, x2, x3):
        x00, (_, _) = self.lstm00(x0)
        x00, (_, _) = self.lstm01(x00)
        x00 = self.SELayer00(x00)
        x00 = x00.reshape(-1, 512)
        x00 = self.fc00(x00)
        out0 = torch.unsqueeze(x00, 0)
        out0 = self.SELayer01(out0)
        out0 = out0.reshape(-1, 256)
        out0 = self.fc01(out0)

        x10, (_, _) = self.lstm10(x1)
        x10, (_, _) = self.lstm11(x10)
        x10 = self.SELayer10(x10)
        x10 = x10.reshape(-1, 512)
        x10 = self.fc10(x10)
        out1 = torch.unsqueeze(x10, 0)
        out1 = self.SELayer11(out1)
        out1 = out1.reshape(-1, 256)
        out1 = self.fc11(out1)

        x20 = torch.unsqueeze(torch.rot90(torch.squeeze(x2, 0)), 0)
        x20 = self.conv1(x20)
        x20 = self.ResNet3(x20)
        x20 = torch.rot90(torch.squeeze(x20, 0), -1)
        out2 = self.fc20(x20)

        x30, (_, _) = self.lstm30(x3)
        x30, (_, _) = self.lstm31(x30)
        x30 = self.SELayer30(x30)
        x30 = x30.reshape(-1, 64)
        x30 = self.fc30(x30)
        out3 = torch.unsqueeze(x30, 0)
        out3 = self.SELayer31(out3)
        out3 = out3.reshape(-1, 32)
        out3 = self.fc31(out3)

        x = torch.cat((x00, x10), 1)
        x = torch.cat((x, x20), 1)
        x = torch.cat((x, x30), 1)

        x = torch.unsqueeze(x, 0)
        x = self.SELayer4(x)
        x = x.reshape(-1, 608)
        x = self.fc4(x)

        x = torch.unsqueeze(x, 0)
        x = self.SELayer5(x)
        x = x.reshape(-1, 256)
        x = self.fc5(x)

        x = torch.unsqueeze(x, 0)
        x = self.SELayer6(x)
        x = x.reshape(-1, 128)
        x = self.fc6(x)
        return out0, out1, out2, out3, x


def BiLSTM_SE_Net():
    model = LSTMMergeSE()
    return model

# input0 = torch.randn(1, 2, 100)
# input1 = torch.randn(1, 2, 100)
# input2 = torch.randn(1, 2, 9)
# input3 = torch.randn(1, 2, 25)
# model = BiLSTM_SE_Net()
# # # print(model)
# out = model(input0, input1, input2, input3)
# print(out)

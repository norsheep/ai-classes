#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Project :back_to_the_realm
@File    :model.py
@Author  :kaiwu
@Date    :2022/11/15 20:57

"""

import torch
import numpy as np
from torch import nn
import torch.nn.functional as F


class Model(nn.Module):

    def __init__(self,
                 state_shape,
                 action_shape=0,
                 softmax=False,
                 dueling=False):
        super().__init__()
        self.dueling = dueling
        cnn_layer1 = [
            nn.Conv2d(4, 16, kernel_size=3, stride=2, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        ]
        cnn_layer2 = [
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        ]
        cnn_layer3 = [
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        ]
        cnn_layer4 = [
            nn.Conv2d(64, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
        ]
        cnn_flatten = [
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True)
        ]

        self.cnn_layer = cnn_layer1 + cnn_layer2 + cnn_layer3 + cnn_layer4 + cnn_flatten
        self.cnn_model = nn.Sequential(*self.cnn_layer)

        fc_layer1 = [
            nn.Linear(np.prod(state_shape), 256),
            nn.ReLU(inplace=True)
        ]
        fc_layer2 = [nn.Linear(256, 128), nn.ReLU(inplace=True)]
        # fc_layer3 = [nn.Linear(128, np.prod(action_shape))]
        # self.fc_layers = fc_layer1 + fc_layer2

        if dueling:
            # 使用dueling dqn
            fc_layer3A = [nn.Linear(128, np.prod(action_shape))]
            fc_layer3V = [nn.Linear(128, np.prod(1))]  # value只对应S，不对应action

            self.fc_A_layers = fc_layer1 + fc_layer2 + fc_layer3A
            self.fc_V_layers = fc_layer1 + fc_layer2 + fc_layer3V

            self.model_A = nn.Sequential(*self.fc_A_layers)
            self.model_V = nn.Sequential(*self.fc_V_layers)
        else:

            fc_layer3 = [nn.Linear(128, np.prod(action_shape))]

            self.fc_layers = fc_layer1 + fc_layer2

            if action_shape:
                self.fc_layers += fc_layer3
            if softmax:
                self.fc_layers += [nn.Softmax(dim=-1)]

            self.model = nn.Sequential(*self.fc_layers)

        # if action_shape:
        #     self.fc_layers += fc_layer3
        # if softmax:
        #     self.fc_layers += [nn.Softmax(dim=-1)]

        # self.model = nn.Sequential(*self.fc_layers)

        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight,
                                    mode="fan_in",
                                    nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight,
                                    mode="fan_in",
                                    nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    # Forward inference
    # 前向推理
    def forward(self, s, state=None, info=None):
        feature_vec, feature_maps = s[0], s[1]
        feature_maps = self.cnn_model(feature_maps)
        feature_maps = feature_maps.view(feature_maps.shape[0], -1)  # 128
        concat_feature = torch.concat([feature_vec, feature_maps],
                                      dim=1)  # 128+404

        if self.dueling:
            A = self.model_A(concat_feature)  # 返回维度为action_shape
            V = self.model_V(concat_feature)  # 返回维度为1
            # print("V:", V)
            # print("A:", A)
            # print("A.mean(1).view(-1,1):", A.mean(1).view(-1,1))
            logits = V + A - A.mean(1).view(-1, 1)  # dim=1处理一个batch
            # print('logits:', logits)
        else:
            logits = self.model(concat_feature)  # relu线性层，返回维度为action_shape

        # logits = self.model(concat_feature)
        return logits, state

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import random
import os, sys
import timm
import torch.nn.functional as F
from glob import glob
from PIL import Image
import cv2
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import albumentations as A
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import math
from collections import OrderedDict
from transformers.models.distilbert.modeling_distilbert import Transformer as T
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
torch.multiprocessing.set_sharing_strategy('file_descriptor')

class Attention(nn.Module):
    def __init__(self, feature_dim, step_dim, bias=True, **kwargs):
        super(Attention, self).__init__(**kwargs)

        self.supports_masking = True

        self.bias = bias
        self.feature_dim = feature_dim
        self.step_dim = step_dim
        self.features_dim = 0

        weight = torch.zeros(feature_dim, 1)
        #         nn.init.kaiming_uniform_(weight)
        self.weight = nn.Parameter(weight)

        if bias:
            self.b = nn.Parameter(torch.zeros(step_dim))

    def forward(self, x, mask=None):
        feature_dim = self.feature_dim
        step_dim = self.step_dim

        eij = torch.mm(
            x.contiguous().view(-1, feature_dim),
            self.weight
        ).view(-1, step_dim)

        if self.bias:
            eij = eij + self.b

        eij = torch.tanh(eij)
        a = torch.exp(eij)

        if mask is not None:
            a = a * mask

        a = a / (torch.sum(a, 1, keepdim=True) + 1e-10)

        weighted_input = x * torch.unsqueeze(a, -1)
        return torch.sum(weighted_input, 1)


class TimmModelCombo(nn.Module):
    def __init__(self, backbone, in_chans, pretrained=False):
        super(TimmModelCombo, self).__init__()

        self.encoder_sagittal = timm.create_model(
            backbone,
            in_chans=2,
            num_classes=1,
            features_only=False,
            drop_rate=0.,
            pretrained=pretrained
        )

        self.encoder_axial = timm.create_model(
            backbone,
            in_chans=1,
            num_classes=1,
            features_only=False,
            drop_rate=0.,
            pretrained=pretrained
        )

        if 'efficient' in backbone:
            hdim = self.encoder_sagittal.conv_head.out_channels
            self.encoder_sagittal.classifier = nn.Identity()
            self.encoder_axial.classifier = nn.Identity()

        elif 'convnext' in backbone:
            hdim = self.encoder_sagittal.head.fc.in_features
            self.encoder_sagittal.head.fc = nn.Identity()
            self.encoder_axial.head.fc = nn.Identity()

        if 'densenet121' in backbone:
            hdim = 1024
            self.encoder_sagittal.classifier = nn.Identity()
            self.encoder_axial.classifier = nn.Identity()

        if 'densenet161' in backbone:
            hdim = 2208
            self.encoder_sagittal.classifier = nn.Identity()
            self.encoder_axial.classifier = nn.Identity()

        if 'densenet201' in backbone:
            hdim = 1920
            self.encoder_sagittal.classifier = nn.Identity()
            self.encoder_axial.classifier = nn.Identity()

        #         self.lstm = nn.LSTM(hdim, 256, num_layers=2, dropout=0., bidirectional=True, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(512, 256),
            nn.Dropout(0.3),
            nn.LeakyReLU(0.1),
            nn.Linear(256, 75),
        )
        self.attention_layer_sagittal = Attention(512, in_chans)
        self.attention_layer_axial = Attention(512, in_chans)
        self.in_chans = in_chans

    def forward(self, x):  # (bs, nslice, ch, sz, sz)
        bs = x.shape[0]
        img_size = x.shape[3]

        x_sagittal = x[:, :, :2, :, :]
        x_axial = x[:, :, 2:3, :, :]

        x_sagittal = x_sagittal.view(bs * self.in_chans, x_sagittal.shape[2], img_size, img_size)
        feat_sagittal = self.encoder_sagittal(x_sagittal)
        feat_sagittal = feat_sagittal.view(bs, self.in_chans, -1)

        x_axial = x_axial.view(bs * self.in_chans, x_axial.shape[2], img_size, img_size)
        feat_axial = self.encoder_axial(x_axial)
        feat_axial = feat_axial.view(bs, self.in_chans, -1)
        #         feat_lstm, _ = self.lstm(feat)
        #         feat_lstm = feat_lstm.contiguous().view(bs * 12, -1)
        #         feat_lstm = self.head(feat_lstm)
        #         feat_lstm = feat_lstm.view(bs, 12, 75).contiguous()
        atten_sagittal = self.attention_layer_sagittal(feat_sagittal)
        atten_axial = self.attention_layer_axial(feat_axial)
        atten = (atten_sagittal + atten_axial) / 2
        out = self.head(atten)
        return out


class TimmModel(nn.Module):
    def __init__(self, backbone, in_chans, pretrained=False):
        super(TimmModel, self).__init__()

        self.encoder = timm.create_model(
            backbone,
            in_chans=2,
            num_classes=1,
            features_only=False,
            drop_rate=0.,
            pretrained=pretrained
        )

        if 'efficient' in backbone:
            hdim = self.encoder.conv_head.out_channels
            self.encoder.classifier = nn.Identity()
        elif 'convnext' in backbone:
            hdim = self.encoder.head.fc.in_features
            self.encoder.head.fc = nn.Identity()

        if 'densenet121' in backbone:
            hdim = 1024
            self.encoder.classifier = nn.Identity()

        if 'densenet161' in backbone:
            hdim = 2208
            self.encoder.classifier = nn.Identity()
        if 'densenet201' in backbone:
            hdim = 1920
            self.encoder.classifier = nn.Identity()

        self.lstm = nn.LSTM(hdim, 256, num_layers=1, dropout=0., bidirectional=True, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(512, 256),
            nn.Dropout(0.3),
            nn.LeakyReLU(0.1),
            nn.Linear(256, 75),
        )
        self.attention_layer = Attention(512, in_chans)
        self.in_chans = in_chans

    def forward(self, x):  # (bs, nslice, ch, sz, sz)
        x = x[:, :, 0:2, :, :]
        bs = x.shape[0]
        img_size = x.shape[3]
        x = x.view(bs * self.in_chans, 2, img_size, img_size)

        feat = self.encoder(x)
        feat = feat.view(bs, self.in_chans, -1)

        #         feat_lstm, _ = self.lstm(feat)

        #         feat_lstm = feat_lstm.contiguous().view(bs * 12, -1)
        #         feat_lstm = self.head(feat_lstm)
        #         feat_lstm = feat_lstm.view(bs, 12, 75).contiguous()
        atten = self.attention_layer(feat)

        out = self.head(atten)
        return out







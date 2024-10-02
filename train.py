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
from dataset import RSNA24Dataset
from models import TimmModel, TimmModelCombo
writer = SummaryWriter()
torch.multiprocessing.set_sharing_strategy('file_descriptor')

rd = 'rsna-2024-lumbar-spine-degenerative-classification'
OUTPUT_DIR = 'rsna-results-2.5d'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
IMG_SIZE = [224, 224]
N_FOLDS = 5
EPOCHS = 100
USE_AMP = True
N_LABELS = 25
N_CLASSES = 3 * N_LABELS
AUG_PROB = 0.75
SELECTED_FOLDS = [2]
SEED = 69
GRAD_ACC = 2
TGT_BATCH_SIZE = 8
IN_CHANS = 12
BATCH_SIZE = TGT_BATCH_SIZE // GRAD_ACC
MAX_GRAD_NORM = None
EARLY_STOPPING_EPOCH = 20
LR = 2e-4 * TGT_BATCH_SIZE / 32
WD = 1e-2
AUG = True
MODEL_NAME = 'convnext_pico.d1_in1k'
# MODEL_NAME = 'convnextv2_pico.fcmae'
NOT_DEBUG = True
N_WORKERS = 4

os.makedirs(OUTPUT_DIR, exist_ok=True)

def set_random_seed(seed: int = 2222, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = deterministic  # type: ignore

set_random_seed(SEED)

df = pd.read_csv(f'{rd}/train.csv')
df = df.fillna(-100)
label2id = {'Normal/Mild': 0, 'Moderate':1, 'Severe':2}
df = df.replace(label2id)

CONDITIONS = [
    'Spinal Canal Stenosis',
    'Left Neural Foraminal Narrowing',
    'Right Neural Foraminal Narrowing',
    'Left Subarticular Stenosis',
    'Right Subarticular Stenosis'
]

LEVELS = [
    'L1/L2',
    'L2/L3',
    'L3/L4',
    'L4/L5',
    'L5/S1',
]
model_names = list(df.columns)[1:]

from pathlib import Path




transforms_train = A.Compose([
    A.OneOf([
        A.MotionBlur(blur_limit=5),
        A.MedianBlur(blur_limit=5),
        A.GaussianBlur(blur_limit=5),
        A.GaussNoise(var_limit=(5.0, 30.0)),
    ], p=AUG_PROB),

    A.OneOf([
        A.OpticalDistortion(distort_limit=1.0),
        A.GridDistortion(num_steps=5, distort_limit=1.),
        A.ElasticTransform(alpha=3),
    ], p=AUG_PROB),

    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=AUG_PROB),
    A.Resize(IMG_SIZE[0], IMG_SIZE[1]),
#     A.CoarseDropout(max_holes=16, max_height=16, max_width=16, min_holes=1, min_height=2, min_width=2, p=AUG_PROB),
    A.Normalize(mean=0.5, std=0.5)
])

transforms_val = A.Compose([
    A.Resize(IMG_SIZE[0], IMG_SIZE[1]),
    A.Normalize(mean=0.5, std=0.5)
])

if not AUG:
    transforms_train = transforms_val

# autocast = torch.cuda.amp.autocast(enabled=USE_AMP, dtype=torch.bfloat16) # if your gpu is newer Ampere, you can use this, lesser appearance of nan than half
autocast = torch.cuda.amp.autocast(enabled=USE_AMP, dtype=torch.half)  # you can use with T4 gpu. or newer
scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP, init_scale=2048)

val_losses = []
train_losses = []
df_tr, df_test = train_test_split(df, test_size=2 / 7, random_state=SEED)
skf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
device = DEVICE

for fold, (trn_idx, val_idx) in enumerate(skf.split(range(len(df)))):
    loss_scale = 1
    if NOT_DEBUG == False:
        if fold == 1: break;
    if fold not in SELECTED_FOLDS:
        print(f"Jump fold {fold}")
        continue;
    else:
        print('#' * 30)
        print(f'Start fold {fold}')
        print('#' * 30)
        print(len(trn_idx), len(val_idx))
        df_train = df.iloc[trn_idx]
        df_valid = df.iloc[val_idx]

        train_ds = RSNA24Dataset(df_train, phase='train', transform=transforms_train)
        train_dl = DataLoader(
            train_ds,
            batch_size=BATCH_SIZE,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
            num_workers=N_WORKERS
        )

        valid_ds = RSNA24Dataset(df_valid, phase='valid', transform=transforms_val)
        valid_dl = DataLoader(
            valid_ds,
            batch_size=BATCH_SIZE * 2,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
            num_workers=0
        )

        #         model = RSNA24Model(MODEL_NAME, IN_CHANS, N_CLASSES, pretrained=True)
        model = TimmModel(MODEL_NAME, IN_CHANS, pretrained=True)

        fname = f'{OUTPUT_DIR}/best_wll_model_fold-{fold}.pt'
        #         if os.path.exists(fname):
        #             model = TimmModel(MODEL_NAME, pretrained=False)
        #             model.load_state_dict(torch.load(fname))
        model.to(device)

        optimizer = AdamW(model.parameters(), lr=LR * 2, weight_decay=WD)
        #         optimizer = torch.optim.SGD(model.parameters(), lr=LR*2, weight_decay=WD, nesterov=True, momentum=0.9)

        warmup_steps = EPOCHS / 10 * len(train_dl) // GRAD_ACC
        num_total_steps = EPOCHS * len(train_dl) // GRAD_ACC
        num_cycles = 0.475
        scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=warmup_steps,
                                                    num_training_steps=num_total_steps,
                                                    num_cycles=num_cycles)
        #         scheduler = get_linear_schedule_with_warmup(optimizer,
        #                                                     num_warmup_steps=warmup_steps,
        #                                                     num_training_steps=num_total_steps)

        weights = torch.tensor([1.0, 2.0, 4.0])
        criterion = nn.CrossEntropyLoss(weight=weights.to(device))
        criterion_cpu = nn.CrossEntropyLoss(weight=weights)
        best_loss = 1.2
        es_step = 0

        for epoch in range(1, EPOCHS + 1):
            print(f'start epoch {epoch}')
            model.train()
            total_loss = 0
            with tqdm(train_dl, leave=True) as pbar:
                optimizer.zero_grad()
                for idx, (x, t) in enumerate(pbar):
                    op = ['nothing', 'nothing', 'nothing', 'nothing', 'nothing']
                    x = x.to(device)
                    t = t.to(device)
                    #                     t = torch.tensor(np.array(one_h(list(t.detach().cpu().numpy())))).to(device)
                    rc = random.sample(op, 1)
                    if rc[0] == 'mixup':
                        x = x.detach().cpu().numpy()
                        t = t.detach().cpu().numpy()
                        reference_data = [{'image': x[i], 'proba': t[i]}
                                          for i in range(len(x))]
                        tr = A.Compose([A.MixUp(reference_data=reference_data,
                                                read_fn=read_fn, p=0.5)])
                        for i in range(len(x)):
                            transformed = tr(image=x[i], global_label=t[i])
                            x[i] = transformed['image']
                            t[i] = transformed['global_label']

                        x = torch.tensor(x).to(device)
                        t = torch.tensor(t).to(device)

                    with autocast:
                        loss = 0
                        y = model(x)
                        for col in range(N_LABELS):
                            pred = y[:, col * 3:col * 3 + 3]
                            gt = t[:, col]
                            loss = loss + loss_scale * criterion(pred, gt) / N_LABELS

                        if not math.isfinite(loss):
                            loss = torch.tensor(1.2 * loss_scale * GRAD_ACC, requires_grad=True)
                        total_loss += loss.item()
                        if GRAD_ACC > 1:
                            loss = loss / GRAD_ACC

                    pbar.set_postfix(
                        OrderedDict(
                            loss=f'{loss.item() * GRAD_ACC:.6f}',
                            lr=f'{optimizer.param_groups[0]["lr"]:.3e}'
                        )
                    )
                    #                     scaler.scale(loss).backward()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM or 1e9)

                    if (idx + 1) % GRAD_ACC == 0:
                        #                         scaler.step(optimizer)
                        #                         scaler.update()
                        optimizer.step()
                        optimizer.zero_grad()
                        if scheduler is not None:
                            scheduler.step()

            train_loss = total_loss / len(train_dl)
            print(f'train_loss:{train_loss / loss_scale:.6f}')
            train_losses.append(train_loss)
            total_loss = 0

            model.eval()
            y_preds, labels = [], []
            with tqdm(valid_dl, leave=True) as pbar:
                with torch.no_grad():
                    for idx, (x, t) in enumerate(pbar):

                        x = x.to(device)
                        t = t.to(device)

                        with autocast:
                            loss = 0
                            loss_ema = 0
                            y = model(x)
                            for col in range(N_LABELS):
                                pred = y[:, col * 3:col * 3 + 3]
                                gt = t[:, col]

                                loss = loss + criterion(pred, gt) / N_LABELS
                                y_pred = pred.float()
                                y_preds.append(y_pred.cpu())
                                labels.append(gt.cpu())

                            if not math.isfinite(loss):
                                loss = torch.tensor(1.2 * loss_scale * GRAD_ACC, requires_grad=True)

                            total_loss += loss.item()

            val_loss = total_loss / len(valid_dl)
            y_preds = torch.cat(y_preds, dim=0)
            print(y_preds.shape)
            labels = torch.cat(labels)

            val_weighted_loss = criterion_cpu(y_preds, labels)
            writer.add_scalar('val_wll', val_weighted_loss, epoch)
            writer.flush()
            print(f'val_loss:{val_loss:.6f}')
            val_losses.append(val_loss)
            if val_weighted_loss < best_loss:

                if device != 'cuda:0':
                    model.to('cuda:0')

                print(f'epoch:{epoch}, best weighted_logloss updated from {best_loss:.6f} to {val_weighted_loss:.6f}')
                best_loss = val_weighted_loss
                fname = f'{OUTPUT_DIR}/best_wll_model_fold-{fold}.pt'
                torch.save(model.state_dict(), fname)
                print(f'{fname} is saved')
                es_step = 0

                if device != 'cuda:0':
                    model.to(device)

            else:
                es_step += 1
                if es_step >= EARLY_STOPPING_EPOCH:
                    print('early stopping')
                    break
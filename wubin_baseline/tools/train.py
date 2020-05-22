
"""
@author: wubin
@connection: 799639771@qq.com
"""

import argparse
import os
import torch
import sys
from torch.backends import cudnn

sys.path.append('.')
from config import cfg
from data import make_data_loader
from modeling import build_model
from loss import make_loss
from solver import make_optimizer, make_optimizer_with_center, WarmupMultiStepLR
from engine import do_train

def train(cfg):
    # prepare datasets
    train_loader, val_loader, num_query, num_classes = make_data_loader(cfg)

    # prepare model
    model = build_model(cfg, num_classes)

    if cfg.MODEL.IF_WITH_CENTER == 'no':
        print("Train without center loss, the loss type is", cfg.MODEL.METRIC_LOSS_TYPE)

        # prepare optimizer
        # optimizer = make_optimizer(cfg, num_classes)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.SOLVER.BASE_LR)
        loss_func = make_loss(cfg, num_classes)

        if cfg.MODEL.PRETRAIN_CHOICE == 'imagenet':
            start_epoch = 0
            scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR,
                                          cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD)
        else:
            print('Only support pretrain choice for imagenet, but got {}', format(cfg.MODEL.PRETRAIN_CHOICE))

        do_train(cfg, model, train_loader, val_loader, optimizer, scheduler, loss_func, num_query, start_epoch)




if __name__=="__main__":
    train(cfg)
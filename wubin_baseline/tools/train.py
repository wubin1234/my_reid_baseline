
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

def train(cfg):
    # prepare datasets
    train_loader, val_loader, query, num_class = make_data_loader(cfg)

if __name__=="__main__":
    train(cfg)
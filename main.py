# -*- coding: utf-8 -*-
"""
@Author: Su Lu

@Date: 2021-04-06 12:19:48
"""

import os
import argparse
import random
import importlib

import numpy as np

import torch
from torch import nn
from torch import optim

from Train import train
from Test import test

def display_args(args):
    print('===== task arguments =====')
    print('data_name = %s' % (args.data_name))
    print('network_name = %s' % (args.network_name))
    print('model_name = %s' % (args.model_name))
    print('N = %d' % (args.N))
    print('K = %d' % (args.K))
    print('Q = %d' % (args.Q))
    print('===== experiment environment arguments =====')
    print('devices = %s' % str(args.devices))
    print('flag_debug = %r' % (args.flag_debug))
    print('flag_no_bar = %r' % (args.flag_no_bar))
    print('n_workers = %d' % (args.n_workers))
    print('===== optimizer arguments =====')
    print('lr_network = %f' % (args.lr_network))
    print('lr = %f' % (args.lr))
    print('point = %s' % str(args.point))
    print('gamma = %f' % (args.gamma))
    print('wd = %f' % (args.wd))
    print('mo = %f' % (args.mo))
    print('===== training procedure arguments =====')
    print('n_training_episodes = %d' % (args.n_training_episodes))
    print('n_validating_episodes = %d' % (args.n_validating_episodes))
    print('n_testing_episodes = %d' % (args.n_testing_episodes))
    print('episode_gap = %d' % (args.episode_gap))
    print('tau = %f' % (args.tau))

if __name__ == '__main__':
    # set random seed
    random.seed(960402)
    np.random.seed(960402)
    torch.manual_seed(960402)
    torch.cuda.manual_seed(960402)
    torch.backends.cudnn.deterministic = True

    # create a parser
    parser = argparse.ArgumentParser()
    # task arguments
    parser.add_argument('--data_name', type=str, default='mini_imagenet', choices=['mini_imagenet', 'tiered_imagenet'])
    parser.add_argument('--network_name', type=str, default='resnet', choices=['resnet'])
    parser.add_argument('--model_name', type=str, default='glofa', choices=['glofa'])
    parser.add_argument('--N', type=int, default=5)
    parser.add_argument('--K', type=int, default=1)
    parser.add_argument('--Q', type=int, default=15)
    # experiment environment arguments
    parser.add_argument('--devices', type=int, nargs='+', default=GV.DEVICES)
    parser.add_argument('--flag_debug', action='store_true', default=False)
    parser.add_argument('--flag_no_bar', action='store_true', default=False)
    parser.add_argument('--n_workers', type=int, default=GV.WORKERS)
    # optimizer arguments
    parser.add_argument('--lr_network', type=float, default=0.0001)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--point', type=int, nargs='+', default=(50,100,150))
    parser.add_argument('--gamma', type=float, default=0.2)
    parser.add_argument('--wd', type=float, default=0.0005)  # weight decay
    parser.add_argument('--mo', type=float, default=0.9)  # momentum
    # training procedure arguments
    parser.add_argument('--n_training_episodes', type=int, default=10000)
    parser.add_argument('--n_validating_episodes', type=int, default=200)
    parser.add_argument('--n_testing_episodes', type=int, default=10000)
    parser.add_argument('--episode_gap', type=int, default=200)
    parser.add_argument('--tau', type=float, default=1) # temperature

    args = parser.parse_args()

    display_args(args)

    # import modules
    Data = importlib.import_module('dataloaders.' + args.data_name)
    Network = importlib.import_module('networks.' + args.student_network_name)
    
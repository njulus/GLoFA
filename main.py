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
from utils import global_variable as GV

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
    print('===== model arguments =====')
    print('tau = %f' % (args.tau))
    print('delta = %f' % (args.delta))

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
    parser.add_argument('--data_name', type=str, default='mini_imagenet', choices=['mini_imagenet'])
    parser.add_argument('--network_name', type=str, default='resnet', choices=['resnet'])
    parser.add_argument('--model_name', type=str, default='glofa', choices=['glofa'])
    parser.add_argument('--N', type=int, default=5)
    parser.add_argument('--K', type=int, default=1)
    parser.add_argument('--Q', type=int, default=15)
    # experiment environment arguments
    parser.add_argument('--devices', type=int, nargs='+', default=GV.DEVICES)
    parser.add_argument('--flag_debug', action='store_true', default=False)
    parser.add_argument('--n_workers', type=int, default=GV.WORKERS)
    # optimizer arguments
    parser.add_argument('--lr_network', type=float, default=0.0001)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--point', type=int, nargs='+', default=(20,30,40))
    parser.add_argument('--gamma', type=float, default=0.2)
    parser.add_argument('--wd', type=float, default=0.0005)  # weight decay
    parser.add_argument('--mo', type=float, default=0.9)  # momentum
    # training procedure arguments
    parser.add_argument('--n_training_episodes', type=int, default=10000)
    parser.add_argument('--n_validating_episodes', type=int, default=200)
    parser.add_argument('--n_testing_episodes', type=int, default=10000)
    parser.add_argument('--episode_gap', type=int, default=200)
    # model arguments
    parser.add_argument('--tau', type=float, default=1)
    parser.add_argument('--delta', type=float, default=1)

    args = parser.parse_args()

    display_args(args)

    data_path = 'datasets/' + args.data_name + '/'

    # import modules
    Data = importlib.import_module('dataloaders.' + args.data_name)
    Network = importlib.import_module('networks.' + args.network_name)
    Model = importlib.import_module('models.' + args.model_name)

    # generate data loaders
    train_data_loader = Data.generate_data_loader(data_path, 'train', args.n_training_episodes, args.N, args.K + args.Q)
    validate_data_loader = Data.generate_data_loader(data_path, 'validate', args.n_validating_episodes, args.N, args.K + args.Q)
    test_data_loader = Data.generate_data_loader(data_path, 'test', args.n_testing_episodes, args.N, args.K + args.Q)
    print('===== data loader ready. =====')

    # generate network
    network = Network.MyNetwork(args)
    if len(args.devices) > 1:
        network = torch.nn.DataParallel(network, device_ids=args.devices)
    print('===== network ready. =====')

    # generate model
    model = Model.MyModel(args, network)
    pretrained_state_dict = torch.load('pretrained_weights.pth')['params']
    pretrained_state_dict = {k:v for k, v in pretrained_state_dict.items() if k.startswith('encoder')}
    model_state_dict = model.state_dict()
    model_state_dict.update(pretrained_state_dict)
    model.load_state_dict(model_state_dict)
    model = model.cuda(args.devices[0])
    print('===== model ready. =====')

    model_save_path = 'saves/trained_models/' + \
                        args.data_name + '_' + args.network_name + '_' + args.model_name + \
                        '_N=' + str(args.N) + \
                        '_K=' + str(args.K) + \
                        '_Q=' + str(args.Q) + \
                        '_lr-net=' + str(args.lr_network) + \
                        '_lr=' + str(args.lr) + \
                        '_point=' + str(args.point) + \
                        '_gamma=' + str(args.gamma) + \
                        '_wd=' + str(args.wd) + \
                        '_mo=' + str(args.mo) + \
                        '_tau=' + str(args.tau) + \
                        '_delta=' + str(args.delta) + \
                        '.model'
    statistic_save_path = 'saves/statistics/' + \
                            args.data_name + '_' + args.network_name + '_' + args.model_name + \
                            '_N=' + str(args.N) + \
                            '_K=' + str(args.K) + \
                            '_Q=' + str(args.Q) + \
                            '_lr-net=' + str(args.lr_network) + \
                            '_lr=' + str(args.lr) + \
                            '_point=' + str(args.point) + \
                            '_gamma=' + str(args.gamma) + \
                            '_wd=' + str(args.wd) + \
                            '_mo=' + str(args.mo) + \
                            '_tau=' + str(args.tau) + \
                            '.stat'

    # create directories
    dirs = os.path.dirname(model_save_path)
    os.makedirs(dirs, exist_ok=True)
    dirs = os.path.dirname(statistic_save_path)
    os.makedirs(dirs, exist_ok=True)

    # training process
    training_loss_list, validating_accuracy_list = train(args, train_data_loader, validate_data_loader, model,
        model_save_path)
    if not args.flag_debug:
        record = {
            'training_loss': training_loss_list,
            'validating_accuracy': validating_accuracy_list
        }
        torch.save(record, statistic_save_path)

    display_args(args)

    # load best model
    if not args.flag_debug:
        record = torch.load(model_save_path)
        model.load_state_dict(record['state_dict'])
        print('best model loaded, validating acc = %f' % record['validating_accuracy'])

    # testing process
    if not args.flag_debug:
        testing_accuracy = test(args, test_data_loader, model)
        print('testing acc = %f' % (testing_accuracy))

# -*- coding: utf-8 -*-
"""
@Author: Su Lu

@Date: 2020-08-01 15:21:24
"""

import os
import random
import pickle
from PIL import Image

import numpy as np

import torch
from torch.utils.data import Dataset
from torch.utils.data import Sampler
from torch.utils.data import DataLoader

from torchvision import transforms

# set random seed
random.seed(960402)
np.random.seed(960402)
torch.manual_seed(960402)
torch.cuda.manual_seed(960402)
torch.backends.cudnn.deterministic = True

class MyDataset(Dataset):
    def __init__(self, data_path, flag_mode):
        super(MyDataset, self).__init__()
        self.data_path = data_path
        self.flag_mode = flag_mode

        if flag_mode == 'train':
            self.csv_path = os.path.join(data_path, 'split/train.csv')
        elif flag_mode == 'validate':
            self.csv_path = os.path.join(data_path, 'split/val.csv')
        elif flag_mode == 'test':
            self.csv_path = os.path.join(data_path, 'split/test.csv')
        else:
            print('Error: flag_mode %s is not supported.' % (flag_mode))
        
        self.data_all, self.label_all = self.read_data()

        self.transform_simple = transforms.Compose([
            transforms.Resize(92),
            transforms.CenterCrop(84),
            transforms.ToTensor(),
            transforms.Normalize(np.array([0.485, 0.456, 0.406]),
                                 np.array([0.229, 0.224, 0.225]))
        ])
        self.transform_augment = transforms.Compose([
            transforms.RandomResizedCrop(84),
            transforms.ColorJitter(brightness = 0.4, contrast = 0.4, saturation = 0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(np.array([0.485, 0.456, 0.406]),
                                 np.array([0.229, 0.224, 0.225]))
        ])

    def read_data(self):
        lines = [x.strip() for x in open(self.csv_path, 'r').readlines()][1:]
        data_all = []
        label_all = []
        name2label = {}
        current_label = 0

        for line in lines:
            file_name, name = line.split(',')
            file_path = self.data_path + 'images/' + file_name
            if name not in name2label.keys():
                name2label[name] = current_label
                current_label += 1
            data_all.append(file_path)
            label_all.append(name2label[name])
        
        return data_all, label_all

    def __len__(self):
        return len(self.label_all)
    
    def __getitem__(self, index):
        path, label = self.data_all[index], self.label_all[index]
        data = Image.open(path).convert('RGB')
        
        if self.flag_mode == 'train':
            image = self.transform_augment(data)
            return image, label
        else:
            image = self.transform_simple(data)
            return image, label



class MySampler(Sampler):
    def __init__(self, label_all, n_episodes, N, S):
        self.label_all = label_all
        self.n_episodes = n_episodes
        self.N = N
        self.S = S

        label_all = np.array(label_all)
        self.class2indexes = []
        for i in range(0, np.max(label_all) + 1):
            indexes = np.argwhere(label_all == i).reshape(-1)
            indexes = torch.from_numpy(indexes)
            self.class2indexes.append(indexes)

    def __len__(self):
        return self.n_episodes
    
    def __iter__(self):
        for i in range(0, self.n_episodes):
            task = []
            classes_needed = torch.randperm(len(self.class2indexes))[:self.N]
            for c in classes_needed:
                indexes = self.class2indexes[c]
                pos = torch.randperm(len(indexes))[:self.S]
                task.append(indexes[pos])
            # task is a Tensor of shape (N, S) before transpose
            # here transpose is important  
            task = torch.stack(task).t().reshape(-1)
            yield task



def generate_data_loader(data_path, flag_mode, n_episodes, N, S):
    my_dataset = MyDataset(data_path, flag_mode)
    my_sampler = MySampler(my_dataset.label_all, n_episodes, N, S)
    my_data_loader = DataLoader(my_dataset, batch_sampler=my_sampler)

    return my_data_loader



# debug test
if __name__ == "__main__":
    data_path = '../datasets/mini_imagenet/'
    task_list_path = data_path + 'task_list/'
    flag_mode = 'train'
    my_dataset = MyDataset(data_path,  flag_mode)

    n_episodes = 10000
    N = 5
    K = 1
    Q = 15
    S = K + Q

    # my_data_loader = generate_data_loader(data_path, flag_mode, n_episodes, N, S)
    # for task_index, task in enumerate(my_data_loader):
    #     image, label = task
    #     print(image.size())
    #     print(label.size())
    #     break
# -*- coding: utf-8 -*-
"""
@Author: Su Lu

@Date: 2021-04-06 19:31:25
"""

import torch
from torch import nn
from torch.nn import functional as F

from networks.set_function import SetFunction

class MyModel(nn.Module):
    def __init__(self, args, network):
        super(MyModel, self).__init__()
        self.args = args
        self.encoder = network
        if args.network_name == 'resnet':
            dimension = 640
        self.f_task = SetFunction(args, input_dimension=dimension, output_dimension=dimension)
        self.f_class = SetFunction(args, input_dimension=dimension, output_dimension=dimension)
        self.h = SetFunction(args, input_dimension=dimension, output_dimension=2)
    
    def forward(self, images):
        embeddings = self.encoder(images) 
        embeddings = embeddings.view(self.args.N * (self.args.K + self.args.Q), -1)

        support_embeddings = embeddings[:self.args.N * self.args.K, :]
        query_embeddings = embeddings[self.args.N * self.args.K:, :]

        mask_task = self.f_task(support_embeddings, level='task').unsqueeze(0)
        mask_class = self.f_class(support_embeddings, level='class').unsqueeze(0)

        alpha = self.h(support_embeddings, level='balance').squeeze(0)
        [alpha_task, alpha_class] = alpha

        masked_support_embeddings = support_embeddings.view(self.args.K, self.args.N, -1) * \
            (1 + mask_task * alpha_task) * (1 + mask_class * alpha_class)
        prototypes = torch.mean(masked_support_embeddings.view(self.args.K, self.args.N, -1), dim=0)
        prototypes = F.normalize(prototypes, dim=1, p=2)

        masked_query_embeddings = query_embeddings.unsqueeze(0).expand(self.args.N, -1, -1) * \
            (1 + mask_task * alpha_task) * (1 + mask_class.transpose(0, 1) * alpha_class)

        logits = torch.bmm(masked_query_embeddings, prototypes.t().unsqueeze(0).expand(self.args.N, -1, -1)) / self.args.tau
        x = torch.arange(self.args.N).long().cuda(self.args.devices[0])
        collapsed_logits = logits[x, :, x].t()

        return collapsed_logits
    
    def get_network_params(self):
        modules = [self.encoder]
        for i in range(len(modules)):
            for j in modules[i].parameters():
                yield j
    
    def get_other_params(self):
        modules = [self.f_task, self.f_class, self.h]
        for i in range(len(modules)):
            for j in modules[i].parameters():
                yield j
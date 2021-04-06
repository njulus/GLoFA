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
        self.f_task = SetFunction()
    
    def forward(self, images):
        embeddings = self.encoder(images) 
        embeddings = embeddings.view(self.args.N * (self.args.K + self.args.Q), -1)

        support_embeddings = embeddings[:self.args.N * self.args.K, :]
        query_embeddings = embeddings[self.args.N * self.args.K:, :]



        prototypes = torch.mean(support_embeddings.view(self.args.K, self.args.N, -1), dim = 0)

        prototypes = F.normalize(prototypes, dim = 1, p = 2)
        logits = torch.mm(query_embeddings, prototypes.t()) / self.args.tau

        return logits
    
    def get_network_params(self):
        modules = [self.encoder]
        for i in range(len(modules)):
            for j in modules[i].parameters():
                yield j
    
    def get_other_params(self):
        modules = []
        for i in range(len(modules)):
            for j in modules[i].parameters():
                yield j
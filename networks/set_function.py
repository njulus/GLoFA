# -*- coding: utf-8 -*-
"""
@Author: Su Lu

@Date: 2020-08-07 14:07:17
"""

import torch
from torch import nn
from torch.nn import functional as F

class SetFunction(nn.Module):
    def __init__(self, args, input_dimension, output_dimension):
        super(SetFunction, self).__init__()
        self.args = args
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension

        self.psi = nn.Sequential(
            nn.Linear(input_dimension, input_dimension  * 2),
            nn.ReLU(),
            nn.Linear(input_dimension * 2, input_dimension * 2),
            nn.ReLU()
        )
        self.rho = nn.Sequential(
            nn.Linear(input_dimension * 3, input_dimension * 2),
            nn.ReLU(),
            nn.Linear(input_dimension * 2, output_dimension),
        )

    def forward(self, support_embeddings, level):
        if level == 'task':
            psi_output = self.psi(support_embeddings)
            rho_input = torch.cat([psi_output, support_embeddings], dim=1)
            rho_input = torch.sum(rho_input, dim=0, keepdim=True)
            rho_output = F.relu6(self.rho(rho_input)) / 6 * self.args.delta
            return rho_output
        elif level == 'class':
            psi_output = self.psi(support_embeddings)
            rho_input = torch.cat([psi_output, support_embeddings], dim=1)
            rho_input = torch.sum(rho_input.view(self.args.K, self.args.N, -1), dim=0)
            rho_output = F.relu6(self.rho(rho_input)) / 6 * self.args.delta
            return rho_output
        elif level == 'balance':
            psi_output = self.psi(support_embeddings)
            rho_input = torch.cat([psi_output, support_embeddings], dim=1)
            rho_input = torch.sum(rho_input, dim=0, keepdim=True)
            rho_output = F.relu(self.rho(rho_input))
            return rho_output
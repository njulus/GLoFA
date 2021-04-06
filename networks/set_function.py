# -*- coding: utf-8 -*-
"""
@Author: Su Lu

@Date: 2020-08-07 14:07:17
"""

import torch
from torch import nn

class SetFunction(nn.Module):
    def __init__(self, input_dimensions, output_dimensions):
        super(SetFunction, self).__init__()
        self.input_dimensions = input_dimensions
        self.output_dimensions = output_dimensions
        
        self.psi = nn.Sequential(
            nn.Linear(input_dimensions, input_dimensions  * 4),
            nn.ReLU(),
            nn.Linear(input_dimensions * 4, input_dimensions),
            nn.ReLU()
        )
        self.rho = nn.Sequential(
            nn.Linear(input_dimensions * 2, input_dimensions * 4),
            nn.ReLU(),
            nn.Linear(input_dimensions * 4, output_dimensions),
            nn.ReLU()
        )

    def forward(self, embeddings, N, K, Q, flag_grain):
        support_embeddings = embeddings[:N * K, :]
        if flag_grain == 'task':
            psi_output = self.psi(support_embeddings)
            rho_input = torch.cat([psi_output, support_embeddings], dim = 1)
            rho_input = torch.sum(rho_input, dim = 0, keepdim = True)
            rho_output = self.rho(rho_input)
            mask = rho_output.expand(N * (K + Q), -1)
            return mask
        elif flag_grain == 'class':
            psi_output = self.psi(support_embeddings)
            rho_input = torch.cat([psi_output, support_embeddings], dim = 1)
            rho_input = torch.sum(rho_input.view(K, N, -1), dim = 0)
            rho_output = self.rho(rho_input)
            rho_output = rho_output.unsqueeze(0)
            rho_output = rho_output.expand(K + Q, N, -1)
            mask = rho_output.reshape(N * (K + Q), -1)
            return mask
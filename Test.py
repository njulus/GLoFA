# -*- coding: utf-8 -*-
"""
@Author: Su Lu

@Date: 2021-04-06 12:19:55
"""

import torch

def test(args, data_loader, model):
    model.eval()
    accuracy = 0
    for task_index, task in enumerate(data_loader):
        images, labels = task
        images = images.float().cuda(args.devices[0])
        labels = labels.long().cuda(args.devices[0])

        logits = model.forward(images)

        query_targets = torch.arange(args.N).repeat(args.Q).long()
        query_targets = query_targets.cuda(args.devices[0])

        predictions = torch.argmax(logits, dim=1)
        accuracy += torch.mean((predictions == query_targets).float()).cpu().item()
    
    accuracy /= len(data_loader)
    return accuracy
# -*- coding: utf-8 -*-
"""
@Author: Su Lu

@Date: 2021-04-06 12:19:53
"""

import torch
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR

from Test import test

def train(args, train_data_loader, validate_data_loader, model, model_save_path):
    optimizer = SGD([
        {'params':model.get_network_params(), 'lr': args.lr_network},
        {'params':model.get_other_params(), 'lr':args.lr}
    ], weight_decay=args.wd, momentum=args.mo, nesterov=True)
    
    scheduler = MultiStepLR(optimizer, args.point, args.gamma)

    training_loss_list = []
    validating_accuracy_list = []
    best_validating_accuracy = 0

    training_loss = 0

    for task_index, task in enumerate(train_data_loader):
        model.train()

        images, labels = task
        images = images.float().cuda(args.devices[0])
        labels = labels.long().cuda(args.devices[0])

        logits = model.forward(images)

        query_targets = torch.arange(args.N).repeat(args.Q).long()
        query_targets = query_targets.cuda(args.devices[0])
        
        loss = nn.CrossEntropyLoss()(logits, query_targets)
        training_loss += loss.cpu().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (task_index + 1) % args.episode_gap == 0:
            training_loss /= args.episode_gap
            validating_accuracy = test(args, validate_data_loader, model)
            training_loss_list.append(training_loss)
            validating_accuracy_list.append(validating_accuracy)
            print('epoch %d finish: training loss = %f, validating acc = %f' % (
                (task_index + 1) / args.episode_gap, training_loss, validating_accuracy
            ))

            if not args.flag_debug:
                if validating_accuracy > best_validating_accuracy:
                    best_validating_accuracy = validating_accuracy
                    record = {
                        'state_dict': model.state_dict(),
                        'validating_accuracy': validating_accuracy,
                        'epoch': (task_index + 1) / args.episode_gap
                    }
                    torch.save(record, model_save_path)
            
            training_loss = 0
            scheduler.step()
    
    return training_loss_list, validating_accuracy_list
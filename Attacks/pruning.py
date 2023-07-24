# pruning

import matplotlib.pyplot as plt
import torch.nn.utils.prune as prune
from utils import *



def prune_model_l1_unstructured(new_model, proportion):
    for name, module in new_model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            print("pruned")
            prune.l1_unstructured(module, name='weight', amount=proportion)
            prune.remove(module, 'weight')
    return new_model

def random_mask(new_model, proportion):
    # maybe add a dimension for the pruning to remove entirely the kernel
    for name, module in new_model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            prune.random_unstructured(module, name='weight', amount=proportion)

    return dict(new_model.named_buffers())

def prune_model_random_unstructured(new_model, proportion):
    dict_mask=random_mask(new_model,proportion)
    for name, module in new_model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            print("pruned")
            weight_name = name + '.weight_mask'
            module.weight = nn.Parameter(module.weight * dict_mask[weight_name])
    return new_model





def train_pruning(net, optimizer, criterion, trainloader, number_epochs, value=None, mask=None):
    # train
    net.train()
    for epoch in range(number_epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):

            # split data into the image and its label
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            if inputs.size()[1] == 1:
                inputs.squeeze_(1)
                inputs = torch.stack([inputs, inputs, inputs], 1)
            # initialise the optimiser
            optimizer.zero_grad()

            # forward
            outputs = net(inputs)
            # backward
            loss = criterion(outputs, labels)
            loss.backward()

            if value != None:
                net = prune_model_l1_unstructured(net, value)
            elif mask != None:
                net = prune_model_random_unstructured(net, mask)

            # update the optimizer
            optimizer.step()
            # loss
            running_loss += loss.item()



'''

    "l1pruning"
    param={"proportion":.99}
    
    "rdnpruning"
    param={"proportion":.99}
'''

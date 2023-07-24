# fine tuning

import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import *

def finetuning(net,epochs,trainloader):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    for epoch in tqdm(range(epochs)):
        net.train()
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # split data into the image and its label
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            # initialise the optimiser
            optimizer.zero_grad()
            # forward
            outputs = net(inputs)
            # backward
            loss = criterion(outputs, labels)
            loss.backward()
            # update the optimizer
            optimizer.step()
            # loss
            running_loss += loss.item()
    return net

'''
    "ft"
    param={"epochs":100,"trainloader":trainloader}
'''



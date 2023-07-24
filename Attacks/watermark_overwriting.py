# This is a sample Python script.
from utils import *


def overwriting(net,NNWmethod,nbr_watermark,watermarking_dict):
    for i in range(nbr_watermark):
        Embeds(watermarking_dict["types"],NNWmethod,net,watermarking_dict)
    return net


def Embeds(types, tools, model, watermarking_dict):
    if types == "1":
        tools.init(model, watermarking_dict)
        trainset, testset, inference_transform = CIFAR10_dataset()
        # hyperparameter of training
        criterion = nn.CrossEntropyLoss()
        num_epochs = 5
        batch_size = 128
        trainloader, testloader = dataloader(trainset, testset, batch_size)

        learning_rate, momentum, weight_decay = 0.01, .9, 5e-4
        optimizer = optim.SGD([
            {'params': model.parameters()}
        ], lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
        model.train()
        epoch = 0
        print("Launching injection.....")
        while epoch < num_epochs:
            print('doing epoch', str(epoch + 1), ".....")
            loss, loss_nn, loss_w = tools.Embedder_one_step(model, trainloader, optimizer, criterion, watermarking_dict)

            loss = (loss * batch_size / len(trainloader.dataset))
            loss_nn = (loss_nn * batch_size / len(trainloader.dataset))
            loss_w = (loss_w * batch_size / len(trainloader.dataset))
            print(' loss  : %.5f   - loss_wm: %.5f, loss_nn: %.5f  ' % (loss, loss_w, loss_nn))
            epoch += 1
    elif types=="0":
        print("Launching injection.....")
        model = tools.Embedder(model, watermarking_dict)
    return model

'''
    "wo"
    param={"NNWmethods":tools,"nbr":2,"watermarking_dict":watermarking_dict,}
'''
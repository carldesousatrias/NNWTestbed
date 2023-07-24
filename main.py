# This is a sample Python script.
from typing import Dict

from utils import *
from Architectures import *
from Attacks.mainAttack import attacks
from NNWmethods import *
import pandas as pd
from torch.nn.utils import prune



def pruning(net, proportion):
    for name, module in net.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=proportion)
            prune.remove(module, 'weight')
    return net


def train(net, trainloader, optimizer, criterion, watermarking_dict=None):
    '''
    :param watermarking_dict: dictionary with all watermarking elements
    :return: the different losses ( global loss, task loss, watermark loss)
    '''
    running_loss = 0
    running_loss_nn = 0
    running_loss_watermark = 0
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
        loss_nn = criterion(outputs, labels)
        # watermark
        loss = loss_nn

        loss.backward()
        # update the optimizer
        optimizer.step()

        # loss
        running_loss += loss.item()
        running_loss_nn += loss_nn.item()
    return running_loss, running_loss_nn, 0

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # (optional) Reproducibility/Repeatability
    # seed=1
    # torch.manual_seed(seed * 3)
    # np.random.seed(seed * 3)

    save = 'vgg16_Tart'
    # initialisation
    num_class = 10
    network = vgg16().to(device)
    # print_net(network)
    criterion = nn.CrossEntropyLoss()
    num_epochs = 200
    batch_size = 128
    learning_rate = 0.01
    trainset, testset, inference_transform = CIFAR10_dataset()
    print(max(trainset.targets))

    # ----------------------------------------------------------------------------------------------------------- #
    # ------------------------------------------Inserter--------------------------------------------------------- #
    # ----------------------------------------------------------------------------------------------------------- #
    ### TBM watermarking
    tools = Tart_tools()
    SF, R, R_stdev = 1, 2, 0.01
    architecture = vgg16()
    lamb = 1e-5
    watermark_obj = 'NNWresources/watermark_example.png'
    watermarking_dict = {'architecture': architecture, 'path_watermark': watermark_obj, 'SF': SF, 'R': R,
                         'std': R_stdev, 'lamb': lamb}

    ### end TBM watermarking
    watermarking_dict = tools.init(network, watermarking_dict)

    if "dataset" in watermarking_dict:
        trainloader, testloader = dataloader(watermarking_dict["dataset"], testset, batch_size)
        print('dataset modified')
    else:
        trainloader, testloader = dataloader(trainset, testset, batch_size)

    optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=80, gamma=.1)
    # training
    network.train()
    for epoch in tqdm(range(num_epochs)):
        loss, _, _ = tools.insertion(network, trainloader, optimizer, criterion, watermarking_dict)
        loss = (loss * batch_size / len(trainloader.dataset))
        print(loss)
        scheduler.step()

    ## Small report
    print('Finished Training:')
    print('Validation error : %.2f' % fulltest(network, testloader))
    print('Extracted watermark : %s  Watermark error : %i' % (tools.detection(network, watermarking_dict)))

    # ----------------------------------------------------------------------------------------------------------- #
    # ------------------------------------------end Inserter----------------------------------------------------- #
    # ----------------------------------------------------------------------------------------------------------- #

    # (optional) save
    if save != '':
        torch.save({
            'model_state_dict': network.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        },  save + '_weights')
        save_water(watermarking_dict, save + '_watermarking_dict.npy')


    # (optional) load
    watermarking_dict = np.load( save + '_watermarking_dict.npy', allow_pickle=True).item()
    checkpoint = torch.load( save + '_weights', map_location=torch.device('cpu'))
    network.load_state_dict(checkpoint["model_state_dict"])

    # ----------------------------------------------------------------------------------------------------------- #
    # ------------------------------------------Attacker--------------------------------------------------------- #
    # ----------------------------------------------------------------------------------------------------------- #
    attackParameter = {'bits': 4}
    network = attacks(network, "quantization", attackParameter)
    # ----------------------------------------------------------------------------------------------------------- #
    # ------------------------------------------end Attacker----------------------------------------------------- #
    # ----------------------------------------------------------------------------------------------------------- #

    # ----------------------------------------------------------------------------------------------------------- #
    # ------------------------------------------Detector--------------------------------------------------------- #
    # ----------------------------------------------------------------------------------------------------------- #
    val_score = fulltest(network, testloader)
    watermark, retrieve_res = tools.detection(network, watermarking_dict)
    print('Validation error : %.2f' % val_score)
    print('Extracted watermark : %s  Watermark error : %2f' % (watermark, retrieve_res))
    # ----------------------------------------------------------------------------------------------------------- #
    # ------------------------------------------end Detector----------------------------------------------------- #
    # ----------------------------------------------------------------------------------------------------------- #

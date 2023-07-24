import torch.nn.functional as F
from utils import *

def distill_unlabeled(y, teacher_scores, T):
    return nn.KLDivLoss()(F.log_softmax(y/T), F.softmax(teacher_scores/T)) * T * T

def test_knowledge_dist(net, water_loss, file_weights, file_watermark, dataset='CIFAR10'):
    epochs_list, test_list, water_test_list = [], [], []

    trainset, testset, _ = CIFAR10_dataset()

    trainloader, testloader = dataloader(trainset, testset, 100)
    student_net = tv.models.vgg16()
    student_net.classifier = nn.Linear(25088, 10)
    student_net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(student_net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    watermarking_dict = np.load(file_watermark, allow_pickle='TRUE').item()
    net.eval()
    for param in net.parameters():
        param.requires_grad = False
    student_net.train()
    for epoch in range(10):
        net.train()
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # split data into the image and its label
            inputs, labels = data
            if dataset == 'MNIST':
                inputs.squeeze_(1)
                inputs = torch.stack([inputs, inputs, inputs], 1)
            inputs = inputs.to(device)
            labels = labels.to(device)

            teacher_output = net(inputs)
            teacher_output = teacher_output.detach()
            _, labels_teacher = torch.max(F.log_softmax(teacher_output, dim=1),dim=1)
            # initialise the optimiser
            optimizer.zero_grad()
            # forward
            outputs = student_net(inputs)
            # backward
            loss = criterion(outputs, labels_teacher)
            loss.backward()
            # update the optimizer
            optimizer.step()
            # loss
            running_loss += loss.item()
        print(running_loss)
    return epochs_list, test_list, water_test_list

def knowledge_distillation(net, epochs, trainloader,student_net):
    student_net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(student_net.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
    net.eval()
    for param in net.parameters():
        param.requires_grad = False
    student_net.train()
    for epoch in range(epochs):
        print('doing epoch', str(epoch + 1), ".....")
        net.train()
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # split data into the image and its label
            inputs, labels = data
            inputs = inputs.to(device)

            teacher_output = net(inputs)
            teacher_output = teacher_output.detach()
            _, labels_teacher = torch.max(F.log_softmax(teacher_output, dim=1), dim=1)
            # initialise the optimiser
            optimizer.zero_grad()
            # forward
            outputs = student_net(inputs)
            # backward
            loss = criterion(outputs, labels_teacher)
            loss.backward()
            # update the optimizer
            optimizer.step()
            # loss
            running_loss += loss.item()
        loss = (running_loss * 128 / len(trainloader.dataset))
        print(' loss  : %.5f   ' % (loss))
    return student_net


'''
    "kd"
    trainset, testset, inference_transform = CIFAR10_dataset()
    trainloader, testloader = dataloader(trainset, testset, 128)
    student = tv.models.vgg16()
    student.classifier = nn.Linear(25088, 10)
    param = {"epochs":5,"trainloader":trainloader,"student":student}
'''
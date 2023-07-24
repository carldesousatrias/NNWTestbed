import torch
import numpy as np
from utils import *



class Merr_tools():
    def __init__(self)-> None:
        super(Merr_tools, self).__init__()

    def to_float(self,x,y):
        return x.type(torch.float) / 255.0, y

    def fast_gradient_signed(self, x,data_grad,epsilon):
        ''' create adversarial examples using the fast gradient method'''
        # Collect the element-wise sign of the data gradient
        sign_data_grad = data_grad.sign()
        # Create the perturbed image by adjusting each pixel of the input image
        perturbed_image = x + epsilon * sign_data_grad
        # Adding clipping to maintain [0,1] range
        return torch.clamp(perturbed_image, 0, 1)

    def gen_adversaries(self,net,criterion,l,dataset,eps):
        ''' generate adversarial datasets to be used as trigger dataset'''
        true_advs=[]
        false_advs=[]
        max_true_advs=max_false_advs=l//2
        for data, target in dataset:
            x = data.to(device)
            y = target.to(device)
            x.requires_grad=True
            y_pred=net(x)
            loss=criterion(y_pred,y)
            net.zero_grad()
            loss.backward()
            data_grad = x.grad.data
            x_advs=self.fast_gradient_signed(x,data_grad,eps)
            x_advs=x_advs.detach()
            y_pred_advs=torch.argmax(net(x_advs),dim=1)
            y_pred = torch.argmax(y_pred, dim=1)
            for i in range(data.size()[0]):
                # x_adv is a true adversary

                if y_pred[i]==y[i] and y_pred_advs[i]!=y[i] and len(true_advs)<max_true_advs:
                    true_advs.append((x_advs[i],y[i]))
                # x_adv is a false adversary
                if y_pred[i]==y[i] and y_pred_advs[i]==y[i] and len(false_advs)<max_false_advs:
                    false_advs.append((x_advs[i],y[i]))
                if len(true_advs)==max_true_advs and len(false_advs)==max_false_advs:
                    return true_advs,false_advs
        return true_advs,false_advs

    def find_tolerance(self,key_length,threshold):
        ''' calculate the tolerance for the verification'''
        theta=0
        factor=2**(-key_length)
        s=0
        while(True):
            # for z in range(theta+1):
            s+=binomial(key_length,theta)
            if factor*s>=threshold:
                return theta
            theta+=1

    def verify(self,net,key_set,threshold=0.05):
        '''verify the watermark based on a treshold'''
        m_k=0
        length=0
        for x,y in key_set:
            length+=len(x)
            preds=torch.argmax(net(x),dim=1)
            y=y.type(torch.int32)
            m_k+=torch.sum(preds!=y)
        theta=self.find_tolerance(length,threshold)
        return {
            'theta':theta,
            'm_k':m_k
        }

    def insertion(self, net, trainloader, optimizer, criterion, watermarking_dict):
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
            # update the optimizer
            optimizer.step()

            # loss
            running_loss += loss.item()

        watermarking_dict['epoch'] = watermarking_dict['epoch'] + 1
        if watermarking_dict['epoch']==watermarking_dict['insertion']:
            print("Stitching frontiers")
            true_advs, false_advs = self.gen_adversaries(net, criterion, watermarking_dict['l'], trainloader, 0.1)
            assert (len(true_advs + false_advs) == watermarking_dict['l'])
            dataset_a = torch.utils.data.DataLoader(true_advs + false_advs, batch_size=watermarking_dict['batch_size'])
            watermarking_dict['key_set'] = dataset_a
            for i in tqdm(range(watermarking_dict['num_epochs'])):
                self.embed(net, dataset_a, optimizer, criterion, watermarking_dict)
        return running_loss, running_loss_nn, running_loss_watermark

    def embed(self, net, trainloader, optimizer, criterion, watermarking_dict):
        '''
        :param watermarking_dict: dictionary with all watermarking elements
        :return: the different losses ( global loss, task loss, watermark loss)
        '''
        running_loss = 0
        net.train()
        for data, target in watermarking_dict["key_set"]:
            data = data.to(device)
            target = target.to(device)
            output = net(data)

            optimizer.zero_grad()
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        return running_loss, running_loss, 0



    def init(self,net,watermarking_dict,save=None):
        '''
        :param net: network
        :param watermarking_dict: dictionary with all watermarking elements
        :param save: file's name to save the watermark
        :return: watermark_dict with a new entry: the secret key matrix X
        '''
        watermarking_dict['epoch'] = 0
        if save != None:
            save_water(watermarking_dict, save)
        return watermarking_dict

    def detection(self,net,watermarking_dict):
        """
        :param file_watermark: file that contain our saved watermark elements
        :return: the extracted watermark, the hamming distance compared to the original watermark
        """
        # watermarking_dict = np.load(file_watermark, allow_pickle='TRUE').item() #retrieve the dictionary
        m_k = 0
        length = 0
        for x, y in watermarking_dict['key_set']:
            length += len(x)
            preds = torch.argmax(net(x), axis=1)
            m_k += torch.sum((preds != y).type(torch.int32))
        theta = self.find_tolerance(length, watermarking_dict['treshold'])
        m_k = m_k.cpu().numpy()
        return  m_k < theta, m_k

'''
    tools = Merr_tools()
    l = 100
    treshold = 0.05
    watermarking_dict = {'l': 100, 'treshold': treshold,"batch_size":batch_size,"num_epoch":num_epochs}

'''
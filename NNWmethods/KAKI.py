from utils import *
import random
import cv2
import matplotlib.pyplot as plt


class Kaki_tools():
    def __init__(self) -> None:
        super(Kaki_tools, self).__init__()

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
            self.embed(net,watermarking_dict)
            print("watermark inserted")


        return running_loss, running_loss_nn, running_loss_watermark

    def detection(self, net, watermarking_dict):
        """
        :param file_watermark: file that contain our saved watermark elements
        :return: the extracted watermark, the hamming distance compared to the original watermark
        """
        N = watermarking_dict["N"]

        for name, parameters in net.named_parameters():
            if watermarking_dict['weight_name'] in name:
                org_shape=parameters.size()
                f_weights = parameters.view(-1,)
                target= f_weights[:N**2]
                target_mat = target.reshape((N,N))
                np_target=target_mat.detach().cpu().numpy()
                # add the secret key here
                np_dct=cv2.dct(np_target)
                #watermarking
                lower = np.fliplr(np.tril(np.fliplr(np_dct), N // 2 - 50))
                middle = np.fliplr(np.tril(np.fliplr(np_dct), N // 2 + 50)) - lower
                min_value, max_value = np.min(middle), np.max(middle)
                all_corr=[]
                middle_retrieved_watermark = middle
                for seed in range(1000):
                    np.random.seed(seed)
                    noise = np.random.uniform(min_value, max_value, np.shape(np_dct))
                    lower_noise = np.fliplr(np.tril(np.fliplr(noise), N // 2 - 50))
                    middle_noise = np.fliplr(np.tril(np.fliplr(noise), N // 2 + 50)) - lower_noise
                    M=np.count_nonzero(middle_noise)
                    corr=(1/M)*np.sum(middle_retrieved_watermark*middle_noise)
                    all_corr.append(corr)
                # plt.plot(all_corr)
                # plt.ylabel("detector response")
                # plt.xlabel("k'")
                # plt.show()
                ## validation
                all_corr=np.abs(all_corr)
                k_max=np.argmax(all_corr)
                k_max_value=np.max(all_corr)
                all_corr2=np.concatenate((all_corr[:k_max],all_corr[k_max+1:]))
                k_max2=np.max(all_corr2)

                return str(k_max), (k_max==watermarking_dict['k'] and k_max_value>3*k_max2)
        return "error weight name", False


    def embed(self, net, watermarking_dict):

        N = watermarking_dict["N"]

        for name, parameters in net.named_parameters():
            if watermarking_dict['weight_name'] in name:
                org_shape=parameters.size()
                f_weights = parameters.view(-1,)
                target= f_weights[:N**2]
                target_mat = target.reshape((N,N))
                np_target=target_mat.detach().cpu().numpy()
                # add the secret key here
                np_dct=cv2.dct(np_target)

                #watermarking
                lower = np.fliplr(np.tril(np.fliplr(np_dct), N // 2 - 50))
                middle = np.fliplr(np.tril(np.fliplr(np_dct), N // 2 + 50)) - lower
                min_value, max_value = np.min(middle), np.max(middle)
                np.random.seed(watermarking_dict['k'])
                noise = np.random.uniform(min_value, max_value, np.shape(np_dct))
                lower_noise = np.fliplr(np.tril(np.fliplr(noise), N // 2 - 50))
                middle_noise = np.fliplr(np.tril(np.fliplr(noise), N // 2 + 50)) - lower_noise
                middle_watermark = middle * (1 + watermarking_dict["alpha"]) - watermarking_dict["alpha"] * middle_noise
                np_dct_watermark = np_dct - middle + middle_watermark
                np_watermark=cv2.idct(np_dct_watermark)
                target_watermark=torch.tensor(np_watermark.reshape(-1),device=device)
                f_weights_watermark=f_weights - torch.concat((target,torch.zeros(f_weights.size()[0]-target.size()[0],
                             device=device))) + torch.concat((target_watermark,torch.zeros(f_weights.size()[0]-target.size()[0],device=device)))
                weights_watermark=torch.reshape(f_weights_watermark,org_shape).float()
                parameters.data=weights_watermark
                watermarking_dict["noise"]=middle_noise
                return net

    def init(self, net, watermarking_dict, save=None):
        '''
        :param net: network
        :param watermarking_dict: dictionary with all watermarking elements
        :param save: file's name to save the watermark
        :return: watermark_dict with a new entry: the secret key matrix X
        '''

        ######################"https://pypi.org/project/pyldpc/
        watermarking_dict['epoch']=0

        if save != None:
            save_water(watermarking_dict, save)
        return watermarking_dict

    def size_of_M(self, net, weight_name):
        '''
        :param net: aimed network
        :param weights_name: aimed layer's name
        :return: the 2nd dimension of the secret key matrix X
        '''
        for name, parameters in net.named_parameters():
            if weight_name in name:
                return parameters.size()[1] * parameters.size()[2] * parameters.size()[3]

    def preprocess(self, net, weights_name, N):
        '''
        :param net: aimed network
        :param weights_name: aimed layer's name
        :return: a vector of dimension CxKxK (flattened weight)
        '''

        for name, parameters in net.named_parameters():
            if weights_name in name:
                f_weights = parameters.view(-1,)
                target= f_weights[:N**2]
                target=target.reshape((N,N))
                np_target=target.detach().cpu().numpy()
                # add the secret key here
                dct=cv2.dct(np_target)
                return dct


    # you can copy-paste this section into main to test Uchida's method
    '''
    tools = Kaki_tools()
    weight_name = 'features.19.weight'
    T = 64
    watermark = torch.tensor(np.random.choice([0, 1], size=(T), p=[1. / 3, 2. / 3]), device=device)
    watermarking_dict = {'N':500,'alpha':0.04, 'k':25, 'weight_name': weight_name, 'watermark': watermark,
                         'epoch':0,'insertion':num_epochs}
    '''
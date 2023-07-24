import random
import math
import cv2
from utils import *
from PIL import Image
from Architectures import *

class Tart_tools():
    def __init__(self) -> None:
        super(Tart_tools, self).__init__()

    def weight_decay(self, net, lamb):
        # add weight decay
        for n, p in net.named_parameters():
            p.grad.data.add_(lamb * p.data)

    def watermark_mask(self, net, P):
        # select watermarked weights
        mask = {}
        for n, p in net.named_parameters():
            mask[n] = torch.rand(p.data.size()) < P
        return mask

    def freeze_watermark(self, net, mask):
        # freeze the watermarked weights
        for n, p in net.named_parameters():
            p.grad.data.mul_(mask[n].type(torch.float))

    def bury_watermark(self, net, raw_watermark, P, SF):
        # place the watermark (RGB image) inside the selected weights
        arch_boundaries = {}
        names = []
        tot_params = []
        for n, p in net.named_parameters():
            arch_boundaries[n] = p.data.numel() - 1
            names.append(n)
            tot_params.append(p.data.numel())
        N = 0
        p = np.zeros(len(names))
        j = 0
        for i in tot_params:
            N += i
            p[j] = i
            j += 1
        watermark = self.watermark_mask(net, P)
        model_dict = net.state_dict()
        ###now generate position and mask
        index = np.zeros([np.size(raw_watermark, 0), np.size(raw_watermark, 1), np.size(raw_watermark, 2), 2], int)
        for i in range(raw_watermark.shape[0]):
            for j in range(raw_watermark.shape[1]):
                for k in range(raw_watermark.shape[2]):
                    index[i, j, k, 0] = np.random.choice(len(names), p=p / N)
                    N -= 1
                    p[index[i, j, k, 0]] -= 1
                    index[i, j, k, 1] = random.randrange(
                        arch_boundaries[names[index[i, j, k, 0]]])  ##parameter index extracted
                    while watermark[names[index[i, j, k, 0]]].view(-1)[
                        index[i, j, k, 1]] == 0:  ##already used, to be resampled
                        index[i, j, k, 1] = random.randrange(
                            arch_boundaries[names[index[i, j, k, 0]]])  ##parameter index extracted
                    model_dict[names[index[i, j, k, 0]]].view(-1)[index[i, j, k, 1]] *= 0
                    model_dict[names[index[i, j, k, 0]]].view(-1)[index[i, j, k, 1]] += (raw_watermark[i, j, k])
                    watermark[names[index[i, j, k, 0]]].view(-1)[index[i, j, k, 1]] = 0  ##mask this value forever
        return index, names, watermark

    def get_watermark(self, path, SF):
        # transform an image into np array between -1 & 1
        im_frame = Image.open(path)
        im_frame = im_frame.convert("RGB")
        width, height = im_frame.size
        np_frame = np.array(im_frame).reshape(height, width, 3)
        np_frame = (np_frame / 255.0 * 2 - 1) * SF
        return np_frame

    def exhume_watermark(self, net, index, names, shap):
        # retrieved the watermaked in the weights
        model_dict = net.state_dict()
        retrieved_watermark = np.ones([np.size(index, 0), np.size(index, 1), np.size(index, 2)], float) * 255
        for i in range(shap[0]):
            for j in range(shap[1]):
                for k in range(shap[2]):
                    retrieved_watermark[i, j, k] = model_dict[names[index[i, j, k, 0]]].view(-1)[index[i, j, k, 1]]
        return retrieved_watermark

    def reinitialize_neighbor(self, net, reference, mask, std,
                              ):
        # reinitialise every weights except watermarked one
        N2pams = reference.state_dict()
        for n, p in net.named_parameters():
            p2 = N2pams[n]
            p.data.copy_(p2.data)
            rev_mask = (-mask[n].type(torch.float) + 1) * torch.randn(mask[n].size(), device=device) * std
            p.data.add_(rev_mask)

    def update_center(self, model, neighbor_models, mask, lamb):
        # all gamma R have a distance to gamma 0 and we update gamma 0 ( diminue the global loss)
        for n, p in model.named_parameters():
            update = torch.zeros(p.data.size(), device=device)
            for rep in range(len(neighbor_models)):
                update += neighbor_models[rep].state_dict()[n].data
            update /= (len(neighbor_models))
            update = update * (mask[n].type(torch.float))
            p.data.add_(-lamb * update)

    def save_retrieved_watermark(self, path, np_frame, SF):
        # transform a np array into an RGB image & save it
        np_frame = (np_frame / SF + 1) / 2.0 * 255
        np_frame = np_frame.astype('uint8')
        im_extracted = Image.fromarray(np_frame)
        im_extracted.save(path, "PNG")

    def pearson(self,original,compressed):
        # compute the pearson correlation coefficient
        original = original.flatten()
        compressed = compressed.flatten()
        pearson=np.corrcoef(original,compressed)
        return pearson[0,1]

    def PSNR(self,original, compressed):
        #calculate the PSNR
        mse = np.mean((original - compressed) ** 2)
        if (mse == 0):  # MSE is zero means no noise is present in the signal .
            # Therefore PSNR have no importance.
            return 1
        max_pixel = 255.0
        psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
        return psnr

    def insertion(self, net, trainloader, optimizer, criterion, watermarking_dict):
        '''
        :param watermarking_dict: dictionary with all watermarking elements
        :return: the different losses ( global loss, task loss, watermark loss)
        '''
        net.train()
        running_loss = 0
        running_loss_wm = 0
        neighbor_models=watermarking_dict['neighbor_models']
        neighbor_optimizer=watermarking_dict['neighbor_optimizer']
        mask=watermarking_dict['mask']
        for data, target in trainloader:
            data = data.to(device)
            target = target.to(device)
            if data.size()[1]   == 1:
                data.squeeze_(1)
                data = torch.stack([data, data, data], 1)
            for model in neighbor_models:
                self.reinitialize_neighbor(model, net, mask, watermarking_dict['std'])

            output = net(data)
            optimizer.zero_grad()
            loss = criterion(output, target)
            loss.backward()
            running_loss += loss.item()

            self.weight_decay(net, 5e-4)
            self.freeze_watermark(net, mask)
            optimizer.step()

            for net_idx in range(len(neighbor_models)):
                output = neighbor_models[net_idx](data)
                neighbor_optimizer[net_idx].zero_grad()
                loss2 = -criterion(output, target)  ###because we MAXIMIZE it!
                loss2.backward()
                self.freeze_watermark(neighbor_models[net_idx], mask)
                neighbor_optimizer[net_idx].step()
                loss2.item()
            running_loss_wm += loss2
            self.update_center(net, neighbor_models, mask, watermarking_dict['lamb'])
        return running_loss,running_loss,running_loss_wm

    def detection(self, net, watermarking_dict):
        """
        :param file_watermark: file that contain our saved watermark elements
        :return: the extracted watermark, the hamming distance compared to the original watermark
        """
        retrieved_watermark = self.exhume_watermark(net, watermarking_dict['index'], watermarking_dict['names'],
                                                          np.shape(watermarking_dict['raw_watermark'], ))
        self.save_retrieved_watermark('exhumed_watermark_temp.png', retrieved_watermark, 1)
        img1 = cv2.imread('exhumed_watermark_temp.png')
        img2 = cv2.imread(watermarking_dict['path_watermark'])
        return 'exhumed_watermark_temp.png',self.pearson(img1, img2)

    def init(self, net, watermarking_dict, save=None):
        '''
        :param net: network
        :param watermarking_dict: dictionary with all watermarking elements
        :param save: file's name to save the watermark
        :return: watermark_dict with a new entry: the secret key matrix X
        '''
        SF=watermarking_dict['SF']
        R=watermarking_dict['R']
        neighbor_models = [watermarking_dict['architecture'].to(device) for i in range(R)]
        raw_watermark = self.get_watermark(watermarking_dict['path_watermark'], SF)
        use_watermark = {}
        index, names, watermark = self.bury_watermark(net, raw_watermark, 1, SF)

        for n, p in net.named_parameters():
            use_watermark[n] = watermark[n].to(device)
        print(len(neighbor_models))
        neighbor_optimizer = [optim.SGD(neighbor_models[i].parameters(), lr=.1) for i in range(R)]
        watermarking_dict['raw_watermark']=raw_watermark
        watermarking_dict['index']=index
        watermarking_dict['names']=names
        watermarking_dict['mask']=use_watermark
        watermarking_dict['neighbor_models']=neighbor_models
        watermarking_dict['neighbor_optimizer']=neighbor_optimizer
        if save!=None:
            save_water(watermarking_dict,save)
        return watermarking_dict

# you can copy-paste this section into main to test Uchida's method
'''
tools=Tart_tools()
SF, R, R_stdev=1,2,0.01
architecture=vgg16()
lamb=1e-5
watermark_obj='NNWresources/watermark_example.png'
watermarking_dict={'architecture':architecture, 'path_watermark':watermark_obj, 'SF':SF,'R':R,'std':R_stdev,'lamb':lamb}

NOTE: weight decay are applied inside the training and should be removed from the optimizer 
'''
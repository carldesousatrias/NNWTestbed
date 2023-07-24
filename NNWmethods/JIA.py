import torch.nn

from utils import *

class Jia_tools():

    def __init__(self):
        super(Jia_tools, self).__init__()

    def list_image(self, main_dir):
        """return all file in the directory"""
        res = []
        for f in os.listdir(main_dir):
            if not f.startswith('.'):
                res.append(f)
        return res

    def add_images(self, dataset, image, label):
        """add an image with its label to the dataset
        :param dataset: aimed dataset to be modified
        :param image: image to be added
        :param label: label of this image
        :return: 0
        """

        (taille, height, width, channel) = np.shape(dataset.data)
        dataset.data = np.append(dataset.data, image)
        dataset.targets.append(label)
        dataset.data = np.reshape(dataset.data, (taille + 1, height, width, channel))
        return 0

    def get_image(self, name):
        """
        :param name: file (including the path) of an image
        :return: a numpy of this image"""
        image = Image.open(name)
        return image

    def pairwise_euclid_distance(self,A):
        '''calculate the pairwise euclidean distance between all the vectors in A'''
        sqr_norm_A = torch.sum(A ** 2, 1).unsqueeze(0)
        sqr_norm_B = torch.sum(A ** 2, 1).unsqueeze(1)
        inner_product = torch.matmul(A, A.transpose(0, 1))
        tile_1 = sqr_norm_A.repeat(A.size()[0],1)
        tile_2 = sqr_norm_B.repeat(1,A.size()[0])
        return tile_1 + tile_2 - 2 * inner_product

    def pairwise_cos_distance(self,A):
        '''calculate the pairwise cosine distance between all the vectors in A'''
        normalized_A = torch.nn.functional.normalize(A,p=2,dim= 1)
        return 1 - torch.matmul(normalized_A, normalized_A.transpose(0, 1))

    def snnl(self, x, y, t, metric='euclidean'):
        '''calculate snnl between x and y'''
        x=x.to(device)
        y=y.to(device)
        Relu = torch.nn.ReLU()
        x = Relu(x)
        same_label_mask = ((y==y.unsqueeze(1)).squeeze()).type(torch.float32)
        if metric == 'euclidean':
            dist = self.pairwise_euclid_distance(torch.reshape(x, [x.size()[0], -1]))
        elif metric == 'cosine':
            dist = self.pairwise_cos_distance(torch.reshape(x, [x.size()[0], -1]))
        else:
            raise NotImplementedError()
        exp = torch.clamp( torch.exp(-(dist / t)) - torch.eye(x.size()[0]).to(device), min=0, max=1)
        prob = (exp / (0.00001 + torch.sum(exp, 1).unsqueeze(1))) * same_label_mask
        loss = - torch.mean(torch.log(0.00001 + torch.sum(prob, 1)))
        return loss

    def watermark_loss(self, net,inputs, watermarking_dict):
        '''
        watermark_loss to embed the watermark
        :param net:
        :param inputs:
        :param watermarking_dict:
        :return:
        '''
        outputs= net(inputs)
        x1,x2,x3=outputs[-2],outputs[-3],outputs[-4]
        inv_temp_1 = 100. / watermarking_dict['temperature'][0]
        inv_temp_2 = 100. / watermarking_dict['temperature'][1]
        inv_temp_3 = 100. / watermarking_dict['temperature'][2]
        w=watermarking_dict['target']
        loss1=self.snnl(x1,w,inv_temp_1)
        loss2=self.snnl(x2,w,inv_temp_2)
        loss3=self.snnl(x3,w,inv_temp_3)

        return watermarking_dict['factor'][0]*loss1+watermarking_dict['factor'][1]*loss2+watermarking_dict['factor'][2]*loss3

    def insertion(self, net, trainloader, optimizer, criterion, watermarking_dict):
        '''
        :param watermarking_dict: dictionary with all watermarking elements
        :return: the different losses ( global loss, task loss, watermark loss)
        '''
        running_loss, running_loss_nn, running_loss_w = 0, 0, 0
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
            ###############################################################################################################################
            loss_w= self.watermark_loss(net,inputs,watermarking_dict)
            loss_nn = criterion(outputs, labels)
            loss = loss_nn + loss_w

            loss.backward()
            watermarking_dict['temp_grad'] = watermarking_dict['temperature'].grad

            # update the optimizer
            optimizer.step()

            # loss
            running_loss += loss.item()
            running_loss_nn += loss_nn.item()
            running_loss_w += loss_w.item()
        watermarking_dict['temperature'].data= watermarking_dict['temperature'].data-  watermarking_dict['lr_temp'] * watermarking_dict['temp_grad']
        return running_loss, running_loss_nn, running_loss_w

    def detection(self, net, watermarking_dict):
        """
        :param file_watermark: file that contain our saved watermark elements
        :return: the extracted watermark, the hamming distance compared to the original watermark
        """
        # watermarking_dict = np.load(file_watermark, allow_pickle='TRUE').item() #retrieve the dictionary
        keys = watermarking_dict['watermark']
        res = 0
        for img_file, label in keys.items():
            img = self.get_image(watermarking_dict['folder'] + img_file)
            net_guess = inference(net, img, watermarking_dict['transform'])
            if net_guess == label:
                res += 1
        return '%i/%i' %(res,len(keys)), len(keys)-res

    def init(self, net, watermarking_dict, save=None):
        '''
        :param net: network
        :param watermarking_dict: dictionary with all watermarking elements
        :param save: file's name to save the watermark
        :return: watermark_dict with a new entry: the secret key matrix X
        '''
        folder = watermarking_dict["folder"]
        list_i = self.list_image(folder)
        keys = {}
        for i in range(len(list_i)):
            keys[list_i[i]] = watermarking_dict['target'].item()

        for img_file, label in keys.items():
            img = self.get_image(folder + img_file)
            for k in range(watermarking_dict["power"]):
                self.add_images(watermarking_dict["dataset"], img, label)
        if save != None:
            save_water(watermarking_dict, save)

        watermarking_dict["watermark"] = keys
        return watermarking_dict

if __name__ == '__main__':
    tools = Jia_tools()
    x = torch.randn(10, 1).to(device)
    y= torch.randn(10, 1).to(device)
    test=tools.snnl(x,y,1)
    print(test)
    '''
    
    !!!!adapted for zha et zho
tools = Jia_tools()
folder = 'NNWresources/zhon/'
target_class =torch.tensor([5])
temperature=torch.tensor([1.,1.,1.],requires_grad=True)
factor=torch.tensor([32,32,32])
lr_temp=torch.tensor([.01,.01,.01])
watermarking_dict = {'folder': folder, 'power': 10, 'dataset': trainset, 'num_class': num_class,
                     'transform': inference_transform, 'target': target_class,'temperature':temperature,
                     'factor':factor,'lr_temp':lr_temp}
    '''
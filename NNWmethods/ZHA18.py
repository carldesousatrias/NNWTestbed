from utils import *
import os
import shutil
import random
from PIL import Image,ImageDraw


class Zha18_tools():
    def __init__(self)-> None:
        super(Zha18_tools, self).__init__()

    def alter_label(self, dataset, index, label):
        """
        change the label of a targeted data
        :param dataset: aimed dataset
        :param index: aimed data
        :param label: the new label
        :return: 0
        """
        dataset.targets[index] = label
        return 0

    def add_images(self, dataset, image, label,watermarking_dict):
        """add an image with its label to the dataset
        :param dataset: aimed dataset to be modified
        :param image: image to be added
        :param label: label of this image
        :return: 0
        """

        if watermarking_dict["flagMNIST"]:
            dataset.data =torch.cat((dataset.data,torch.tensor(image).unsqueeze(0)),0)
            test1,test2=dataset.targets, torch.tensor(label).unsqueeze(-1)
            dataset.targets = torch.cat((test1,test2), 0).squeeze()
        else:
            (taille, height, width, channel) = np.shape(dataset.data)
            dataset.data = np.append(dataset.data, image)
            dataset.targets.append(label)
            dataset.data = np.reshape(dataset.data, (taille + 1, height, width, channel))
        return 0

    def new_label(self, dataset, index, num_class=10):
        """
        select a random label different from the original one
        :param dataset: aimed dataset
        :param index: aimed data
        :param num_class: total number of classes
        :return: the new label
        """
        pioche = np.arange(num_class)
        pioche = np.delete(pioche, dataset.targets[index])
        return np.random.choice(pioche)

    def get_image(self, name):
        """
        :param name: file (including the path) of an image
        :return: a numpy of this image
        """
        image = Image.open(name)
        return image

    def create_watermark(self, dataset, size, num_class):
        """create a dict that represent the watermark
        :param dataset: aimed dataset
        :param indexes: selected datas' indexes
        :return: the watermark
        """
        indexes = random.sample(range(len(dataset)), size)
        watermark = {}
        for indx in indexes:
            watermark[indx] = self.new_label(dataset, indx, num_class)
        return watermark

    def water_image(self, dataset, index, method):
        """ return a watermarked image based on one of the 3 methods
        :param dataset: aimed dataset
        :param index: data's index
        :param method: 'visible/noise/invisible'
        :return: modified image
        """
        img = dataset.data[index]
        img = np.array(img)
        if method == 'visible':
            new_img = Image.fromarray(img)
            d = ImageDraw.Draw(new_img)
            d.text((0, 0), "wm", fill=(255))
            return np.array(new_img)
        elif method == 'noise':
            np.random.seed(0)
            noise = np.random.randint(32, size=np.shape(img))
            new_img = np.zeros(np.shape(img), dtype=int)
            taille = np.shape(img)
            for i in range(taille[0]):
                for j in range(taille[1]):
                    for k in range(taille[2]):
                        new_img[i, j, k] = img[i, j, k] + noise[i, j, k]
                        if new_img[i, j, k] > 255:
                            new_img[i, j, k] = 255
            return np.array(new_img)
        elif method == 'invisible':
            new_img = Image.fromarray(img).convert("RGBA")
            txt = Image.new('RGBA', new_img.size, (255, 255, 255, 0))
            d = ImageDraw.Draw(txt)
            d.text((0, 0), "wm", (255, 255, 255, 10))
            im_water = Image.alpha_composite(new_img, txt).convert("RGB")
            return np.array(im_water)

    def insertion(self, net, trainloader, optimizer, criterion, watermarking_dict):
        '''
        :param watermarking_dict: dictionary with all watermarking elements
        :return: the different losses ( global loss, task loss, watermark loss)
        '''
        running_loss = 0
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

        return running_loss, running_loss, 0

    def detection(self, net, watermarking_dict):
        """
        :param file_watermark: file that contain our saved watermark elements
        :return: the extracted watermark, the hamming distance compared to the original watermark
        """
        # watermarking_dict = np.load(file_watermark, allow_pickle='TRUE').item() #retrieve the dictionary
        keys = watermarking_dict['watermark']
        res = 0
        for img_file, label in keys.items():
            img = self.get_image(watermarking_dict['folder'] + str(img_file)+'.png')
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
        method=watermarking_dict["method"]
        shutil.rmtree(folder)
        os.makedirs(folder,exist_ok=True)
        dataset=watermarking_dict["dataset"]
        keys = self.create_watermark(dataset, watermarking_dict["size"], watermarking_dict["num_class"])

        for indx, label in keys.items():
            #modification
            np_img=self.water_image(dataset,indx,method)
            if method != None:
                if watermarking_dict["flagMNIST"]:
                    dataset.data[indx] = torch.tensor(np_img)
                else:
                    dataset.data[indx] = np_img
            self.alter_label(dataset, indx, label)
            # power
            for k in range(watermarking_dict["power"]):
                self.add_images(watermarking_dict["dataset"], np_img, label, watermarking_dict)
            #save
            img = Image.fromarray(np_img)
            img.save(folder+str(indx)+".png")

        watermarking_dict["watermark"] = keys
        if save != None:
            save_water(watermarking_dict, save)


        return watermarking_dict

'''
tools=Zha18_tools()
folder = 'NNWresources/zha/'
power=10
size=26
method='visible'
watermarking_dict = {'folder': folder,'size':size,'power':power, 'method':method, 'dataset': trainset,
 'num_class': num_class, 'transform':inference_transform, "flagMNIST":False}

'''
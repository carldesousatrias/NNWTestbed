from utils import *
import os
from PIL import Image



class Adi_tools():
    def __init__(self)-> None:
        super(Adi_tools, self).__init__()

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
        return np.array(image)

    def insertion(self, net, trainloader, optimizer, criterion, watermarking_dict):
        '''
        :param watermarking_dict: dictionary with all watermarking elements
        :return: the different losses ( global loss, task loss, watermark loss)
        '''
        running_loss = 0
        wmloader=watermarking_dict['wmloader']
        wminputs, wmtargets = [], []
        if wmloader:
            for wm_idx, (wminput, wmtarget) in enumerate(wmloader):
                wminput, wmtarget = wminput.to(device), wmtarget.to(device)
                wminputs.append(wminput)
                wmtargets.append(wmtarget)

            # the wm_idx to start from
            wm_idx = np.random.randint(len(wminputs))
        for i, data in enumerate(trainloader, 0):
            # split data into the image and its label
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            if wmloader:
                inputs = torch.cat([inputs, wminputs[(wm_idx + i) % len(wminputs)]], dim=0)
                labels = torch.cat([labels, wmtargets[(wm_idx + i) % len(wminputs)]], dim=0)

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
        wmloader= watermarking_dict['wmloader']
        net.eval()
        res = 0
        total = 0
        for i, data in enumerate(wmloader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = net(inputs)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            res += predicted.eq(labels.data).cpu().sum()

        return '%i/%i' %(int(res),total), total-res

    def init(self, net, watermarking_dict, save=None):
        '''
        :param net: network
        :param watermarking_dict: dictionary with all watermarking elements
        :param save: file's name to save the watermark
        :return: watermark_dict with a new entry: the secret key matrix X
        '''
        folder=watermarking_dict["folder"]
        for elmnt in os.listdir(folder):
            if ".txt" in elmnt:labels_path=elmnt
        wmset = ImageFolderCustomClass(
            folder,
            watermarking_dict["transforms"])
        img_nlbl = []
        wm_targets = np.loadtxt(os.path.join(folder, labels_path))
        for idx, (path, target) in enumerate(wmset.imgs):
            img_nlbl.append((path, int(wm_targets[idx])))
        wmset.imgs = img_nlbl

        wmloader = torch.utils.data.DataLoader(
            wmset, batch_size=watermarking_dict["batch_size"], shuffle=True,
            num_workers=4, pin_memory=True)
        watermarking_dict['wmloader']=wmloader

        if save != None:
            save_water(watermarking_dict,save)


        return watermarking_dict

'''
tools = Adi_tools()
folder = 'NNWresources/adi/'
watermarking_dict = {'folder': folder, 'batch_size':batch_size, 'transforms': inference_transform}
'''
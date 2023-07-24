import torch.nn.functional

from utils import *
import torch.nn.functional as F


class Rouh_tools():
    def __init__(self)-> None:
        super(Rouh_tools, self).__init__()

    def small_net(self,net, weights_name):
        '''find the position of the layer with the name_w in net.modules'''
        i = 1
        for name, parameters in net.named_parameters():
            i += 1
            if weights_name in name:
                return i
        return "error"

    def hooked_net(self, net, weight_name):
        '''return the network from the first layer to the aimed layer '''
        hook = self.small_net(net, weight_name)
        test = list(net.children())[0]
        # print(test)
        new_model = nn.Sequential(*list(net.children())[0][:hook]).to(device)
        return new_model

    def add_centers(self,net,weights_name,centers):
        ''' add centers to the aimed layer'''
        for name, parameters in net.named_parameters():
            if weights_name in name:
                parameters.data.add_(centers)
                return

    def subsample_training_data(self,dataset, target_class):
        # train_indices = []
        # for i in range(len(dataset.targets)):
        #     if dataset.targets[i] == target_class:
        #         train_indices.append(i)
        train_indices = (np.array(dataset.targets) == target_class).nonzero()[0]  # .reshape(-1)
        subsample_len = int(np.floor(0.5 * len(train_indices)))
        subset_idx = np.random.randint(len(train_indices), size=subsample_len)
        train_subset = torch.utils.data.Subset(dataset, train_indices[subset_idx])
        dataloader = torch.utils.data.DataLoader(train_subset, batch_size=128, shuffle=False)
        return dataloader

    def get_activation(self,net,train_subset_loader):
        '''return the activation of the aimed layer'''
        activations = []
        net.eval()
        with torch.no_grad():
            for d, t in train_subset_loader:
                d = d.to(device)
                _, feat = net(d)
                activations.extend(feat.detach().cpu().numpy())
        return np.stack(activations, 0)

    def extraction(self, net, watermarking_dict):
        """
        :param net: aimed network
        :param weight_name: aimed layer's name
        :param X: secret key matrix
        :return: the extracted watermark
        """
        watermark=watermarking_dict['watermark'][:, watermarking_dict['target_class']]
        X = watermarking_dict['X'].detach().cpu().numpy()
        activ_class_K= self.get_activation(net,watermarking_dict["train_subset_loader"])
        activ_centerK = np.mean(activ_class_K, axis=0)
        X_Ck = np.dot(X, activ_centerK)
        extract = 1 / (1 + np.exp(-X_Ck))
        return extract, watermark

    def hamming(self, s1, s2):
        '''
        :param s1: sequence 1
        :param s2: sequence 2
        :return: the hamming distance between 2 vectors
        '''
        assert len(s1) == len(s2)
        return sum(c1 != c2 for c1, c2 in zip(s1, s2))

    def compute_BER(self,decode_wmark, b_classK):
        b_classK = np.reshape(b_classK, (-1, 1))
        diff = np.abs(decode_wmark - b_classK)
        BER = np.sum(diff) / b_classK.size
        return BER

    def loss(self, net, weight_name, inputs, labels, X, watermark, centers, target_class):
        '''
        :param net: aimed network
        :param weight_name: aimed layer's name
        :param X: secret key matrix
        :param watermark: the watermark
        :param target_class: the targeted class
        :return: Uchida's loss
        '''
        _, feat = net(inputs)
        centers_batch=torch.gather(centers, 0, labels.unsqueeze(1).repeat(1, feat.shape[1]))
        loss1=F.mse_loss(feat,centers_batch,reduction='sum') /2
        centers_batch_reshape=centers_batch.unsqueeze(1)
        centers_reshape=centers.unsqueeze(0)
        pairwise_dists = (centers_batch_reshape-centers_reshape)**2
        pairwise_dists=torch.sum(pairwise_dists,dim=-1)
        arg= torch.topk(-pairwise_dists,k=2)[1]
        arg = arg[:, -1]
        closest_cents = torch.gather(centers, 0, arg.unsqueeze(1).repeat(1, feat.shape[1]))
        dists=torch.sum((centers_batch-closest_cents)**2,dim=-1)
        cosines=torch.mul(closest_cents,centers_batch)
        cosines= torch.sum(cosines,dim=-1)
        loss2=cosines*dists-dists
        loss2=torch.mean(loss2)
        loss3=torch.sum(torch.abs(1-torch.sum(centers**2,dim=1)))

        return loss1+loss2+loss3



    def loss4(self, net, inputs, labels, watermarking_dict):

        X=watermarking_dict['X']
        watermark=watermarking_dict['watermark']
        _, feats = net(inputs)
        centers_batch = torch.gather(watermarking_dict["centers"], 0, labels.unsqueeze(1).repeat(1, feats.shape[1]))
        embed_center_idx = watermarking_dict["target_class"]
        idx_classK = (labels == embed_center_idx).nonzero()[0]
        activ_classK = torch.gather(centers_batch, 0,
                                    idx_classK.unsqueeze(1).repeat(1, feats.shape[1]))
        center_classK = torch.mean(activ_classK, dim=0)
        Xc = torch.matmul(X, center_classK)
        bk = watermark[:, embed_center_idx]
        probs = torch.sigmoid(Xc)
        bk_float = bk*1.
        Binary_loss = nn.BCELoss()
        loss4 = Binary_loss(input=probs, target=bk_float)
        return loss4

    def size_of_M(self, net, weight_name):
        '''
        :param net: aimed network
        :param weights_name: aimed layer's name
        :return: the 2nd dimension of the secret key matrix X
        '''

        x=torch.randn((10,3,32,32),device=device)
        outputs,feat = net.forward(x)
        return feat.size()[1]

    def insertion(self, net, trainloader, optimizer, criterion, watermarking_dict):
        '''
        :param watermarking_dict: dictionary with all watermarking elements
        :return: the different losses ( global loss, task loss, watermark loss)
        '''
        running_loss = 0
        running_loss_nn = 0
        running_loss_watermark = 0
        Binary_loss = nn.BCELoss()
        X=watermarking_dict['X']
        watermark=watermarking_dict['watermark']
        centers = watermarking_dict["centers"]
        center_optimizer = watermarking_dict["centers_optimizer"]
        net.train()
        for i, data in enumerate(trainloader, 0):
            # split data into the image and its label
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # initialise the optimiser
            optimizer.zero_grad()
            center_optimizer.zero_grad()
            # forward
            outputs, feats = net(inputs)

            # backward
            loss_nn = criterion(outputs, labels)
            # watermark
            centers_batch = torch.gather(centers, 0, labels.unsqueeze(1).repeat(1, feats.shape[1]))
            loss1 = F.mse_loss(feats, centers_batch, reduction='sum') / 2
            centers_batch_reshape = centers_batch.unsqueeze(1)
            centers_reshape = centers.unsqueeze(0)
            pairwise_dists = (centers_batch_reshape - centers_reshape) ** 2
            pairwise_dists = torch.sum(pairwise_dists, dim=-1)
            arg = torch.topk(-pairwise_dists, k=2)[1]
            arg = arg[:, -1]
            closest_cents = torch.gather(centers, 0, arg.unsqueeze(1).repeat(1, feats.shape[1]))
            dists = torch.sum((centers_batch - closest_cents) ** 2, dim=-1)
            cosines = torch.mul(closest_cents, centers_batch)
            cosines = torch.sum(cosines, dim=-1)
            loss2 = cosines * dists - dists
            loss2 = torch.mean(loss2)
            loss3 = torch.sum(torch.abs(1 - torch.sum(centers ** 2, dim=1)))
            loss_power=loss1+loss2+loss3

            embed_center_idx = watermarking_dict["target_class"]
            activ_classK = feats[labels == embed_center_idx]
            center_classK = torch.mean(activ_classK, dim=0)
            Xc = torch.matmul(X, center_classK)
            bk = watermark[:, embed_center_idx]
            probs = torch.sigmoid(Xc)
            probs = torch.clamp(probs,min=1e-2,max=1-1e-2)
            bk_float=bk * 1.
            # print(probs)
            loss_watermark = Binary_loss(input=probs, target=bk_float)
            # print(loss_watermark)

            loss = loss_nn + watermarking_dict['scale']*loss_power + watermarking_dict['gamma2'] * loss_watermark

            loss.backward()
            # update the optimizer
            optimizer.step()
            center_optimizer.step()
            # loss
            running_loss += loss.item()
            running_loss_nn += loss_nn.item()
            running_loss_watermark += loss_watermark.item()
        watermarking_dict["centers"] = centers
        watermarking_dict["centers_optimizer"]=center_optimizer
        return running_loss, running_loss_nn, running_loss_watermark

    def detection(self, net, watermarking_dict):
        """
        :param file_watermark: file that contain our saved watermark elements
        :return: the extracted watermark, the hamming distance compared to the original watermark
        """
        # watermarking_dict = np.load(file_watermark, allow_pickle='TRUE').item() #retrieve the dictionary

        extraction, watermark = self.extraction(net, watermarking_dict)
        extraction_r = (extraction > 0.5) * 1  # <.5 = 0 and >.5 = 1
        print(extraction_r, watermark)
        res = self.hamming(watermark, extraction_r)/len(watermark)
        return extraction, float(res)*100

    def init(self, net, watermarking_dict, save=None):
        '''
        :param net: network
        :param watermarking_dict: dictionary with all watermarking elements
        :param save: file's name to save the watermark
        :return: watermark_dict with a new entry: the secret key matrix X
        '''
        M = self.size_of_M(net, watermarking_dict['weight_name'])
        T = len(watermarking_dict['watermark'])
        X = torch.randn((T, M), device=device,dtype=torch.float32)
        watermarking_dict['X'] = X
        x_train_subset_loader = self.subsample_training_data(watermarking_dict["train_subset_loader"],
                                                             watermarking_dict["target_class"])
        watermarking_dict["train_subset_loader"] = x_train_subset_loader
        # self.add_centers(net,watermarking_dict['weight_name'],watermarking_dict['centers'])
        if save != None:
            save_water(watermarking_dict, save)
        return watermarking_dict
'''
    
    network = ROUH_xxxx0.to(device)
    
    [.....]
    
    ### TBM watermarking
    tools = Rouh_tools()
    scale = 1e-4  # for loss1
    gamma2 = 0.01  # for loss2
    weight_name = 'features.18.0.w'  # target layer to carry WM
    embed_bits = 16
    target_class = 0
    watermark = torch.tensor(np.random.randint(2, size=(embed_bits, num_class)), device=device)
    centers = torch.nn.Parameter(torch.rand(num_class, 512).to(device), requires_grad=True).to(device)
    centers_optimizer = optim.SGD([centers], lr=learning_rate, momentum=.9)
    watermarking_dict = {'lambd': 0.1, 'weight_name': weight_name, 'watermark': watermark, 'target_class': target_class,
                         'scale': scale, 'gamma2': gamma2, 'centers': centers, 'centers_optimizer': centers_optimizer,
                         "train_subset_loader": trainset}
'''
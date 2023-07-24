from utils import *


class Uchi_tools():
    def __init__(self) -> None:
        super(Uchi_tools, self).__init__()

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
            loss_nn = criterion(outputs, labels)
            # watermark
            loss_watermark = self.loss(net, watermarking_dict['weight_name'], watermarking_dict['X'], watermarking_dict['watermark'])

            loss = loss_nn + watermarking_dict['lambd'] * loss_watermark  # Uchida

            loss.backward()
            # update the optimizer
            optimizer.step()

            # loss
            running_loss += loss.item()
            running_loss_nn += loss_nn.item()
            running_loss_watermark += loss_watermark.item()
        return running_loss, running_loss_nn, running_loss_watermark

    def detection(self, net, watermarking_dict):
        """
        :param file_watermark: file that contain our saved watermark elements
        :return: the extracted watermark, the hamming distance compared to the original watermark
        """
        # watermarking_dict = np.load(file_watermark, allow_pickle='TRUE').item() #retrieve the dictionary
        watermark = watermarking_dict['watermark'].to(device)
        X = watermarking_dict['X'].to(device)
        weight_name = watermarking_dict["weight_name"]
        extraction = self.extraction(net, weight_name, X)
        extraction_r = torch.round(extraction) # <.5 = 0 and >.5 = 1
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
        X = torch.randn((T, M), device=device)
        watermarking_dict['X']=X
        if save != None:
            save_water(watermarking_dict,save)
        return watermarking_dict

    def projection(self, X, w):
        '''
        :param X: secret key matrix
        :param w: flattened weight
        :return: sigmoid of the matrix multiplication of the 2 inputs
        '''
        sigmoid_func = nn.Sigmoid()
        res = torch.matmul(X, w)
        sigmoid = sigmoid_func(res)
        return sigmoid

    def flattened_weight(self, net, weights_name):
        '''
        :param net: aimed network
        :param weights_name: aimed layer's name
        :return: a vector of dimension CxKxK (flattened weight)
        '''

        for name, parameters in net.named_parameters():
            if weights_name in name:
                f_weights = torch.mean(parameters, dim=0)
                f_weights = f_weights.view(-1, )
                return f_weights

    def extraction(self, net, weights_name, X):
        '''
        :param net: aimed network
        :param weights_name: aimed layer's name
        :param X: secret key matrix
        :return: a binary vector (watermark)
        '''
        W = self.flattened_weight(net, weights_name)
        return self.projection(X, W)

    def hamming(self, s1,s2):
        '''
        :param s1: sequence 1
        :param s2: sequence 2
        :return: the hamming distance between 2 vectors
        '''
        assert len(s1) == len(s2)
        return sum(c1 != c2 for c1, c2 in zip(s1, s2))

    def loss(self, net, weights_name, X, watermark):
        '''
        :param net: aimed network
        :param weights_name: aimed layer's name
        :param X: secret key matrix
        :param watermark: the watermark
        :return: Uchida's loss
        '''
        loss = 0
        W = self.flattened_weight(net, weights_name)
        yj = self.projection(X, W)
        for i in range(len(watermark)):
            loss += watermark[i] * torch.log2(yj[i]) + (1 - watermark[i]) * torch.log2(1 - yj[i])
        return -loss/len(watermark)

    def size_of_M(self, net, weight_name):
        '''
        :param net: aimed network
        :param weights_name: aimed layer's name
        :return: the 2nd dimension of the secret key matrix X
        '''
        for name, parameters in net.named_parameters():
            if weight_name in name:
                return parameters.size()[1] * parameters.size()[2] * parameters.size()[3]




    # you can copy-paste this section into main to test Uchida's method
    '''
    tools=Uchi_tools()
    weight_name = 'features.19.weight'
    T = 64
    watermark = torch.tensor(np.random.choice([0, 1], size=(T), p=[1. / 3, 2. / 3]), device=device)
    watermarking_dict={'lambd':0.1, 'weight_name':weight_name,'watermark':watermark}
    '''
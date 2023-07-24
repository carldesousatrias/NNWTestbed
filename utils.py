# import nécessaire à tous
import torch
import os
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision as tv
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# functions
def binomial(n, k):
    if not 0 <= k <= n:
        return 0
    b = 1
    for t in range(min(k, n-k)):
        b *= n
        b //= t+1
        n -= 1
    return b

def print_net(net):
    for name, parameters in net.named_parameters():
        print(name, ':', parameters.size())
    return

def save_water( watermarking_dict, file_watermark):
    """
    save the watermarking_dict in file watermark
    """
    np.save(file_watermark, watermarking_dict)

def inference(net, img, transform):
    """make the inference for one image and a given transform"""
    img_tensor= transform(img).unsqueeze(0)
    net.eval()
    with torch.no_grad():
        logits = net.forward(img_tensor.to(device))
        _, predicted = torch.max(logits, 1) # take the maximum value of the last layer
    return predicted

def fulltest(net,testloader):
    # test complet
    correct = 0
    total = 0

    # torch.no_grad do not train the network
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            if inputs.size()[1]   == 1:
                inputs.squeeze_(1)
                inputs = torch.stack([inputs, inputs, inputs], 1)
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = net(inputs)
            if len(outputs) ==2:outputs,_=outputs
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
    return 100 - (100 * float(correct) / total)

def dataloader(trainset,testset,batch_size=100):
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2)

    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2)

    return trainloader,testloader

def CIFAR10_dataset():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # datasets
    trainset = tv.datasets.CIFAR10(
        root='./data/',
        train=True,
        download=True,
        transform=transform_train)

    testset = tv.datasets.CIFAR10(
        './data/',
        train=False,
        download=True,
        transform=transform_test)

    return trainset, testset, transform_test

def CIFAR100_dataset():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # datasets
    trainset = tv.datasets.CIFAR100(
        root='./data/',
        train=True,
        download=True,
        transform=transform_train)

    testset = tv.datasets.CIFAR100(
        './data/',
        train=False,
        download=True,
        transform=transform_test)

    return trainset, testset, transform_test

def MNIST_dataset():
    transform_train = transforms.Compose([
        transforms.Grayscale(3),
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),])

    transform_test = transforms.Compose([
        transforms.Grayscale(3),
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),])

    # datasets
    trainset = tv.datasets.FashionMNIST(
        root='./data/',
        train=True,
        download=True,
        transform=transform_train)

    testset = tv.datasets.FashionMNIST(
        './data/',
        train=False,
        download=True,
        transform=transform_test)

    return trainset, testset, transform_test

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')
            # return img.convert('L')
            # return img


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']


def is_image_file(filename):
    """Checks if a file is an image.
    Args:
        filename (string): path to a file
    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, class_to_idx):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class ImageFolderCustomClass(torch.utils.data.Dataset):
    """A generic data loader where the images are arranged in this way: ::
        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png
        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png
    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader, custom_class_to_idx=None) :
        if custom_class_to_idx is None:
            classes, class_to_idx = find_classes(root)
        else:
            class_to_idx = custom_class_to_idx
            classes = list(class_to_idx.keys())
        imgs = make_dataset(root, class_to_idx)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp,
                                     self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp,
                                   self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

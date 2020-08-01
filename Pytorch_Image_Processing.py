from __future__ import print_function, division
import torch
import gzip
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
from torch.autograd import Variable
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision
import pandas as pd
import numpy as np
from scipy import ndimage, misc
import h5py
import pickle
import tensorflow as tf
import cv2


class Net(nn.Module):#nn with 2 conv and 2 fc
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 100, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),nn.BatchNorm2d(100))
        self.layer2 = nn.Sequential(
            nn.Conv2d(100, 220, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),nn.BatchNorm2d(220))
        self.layer3 = nn.Sequential(
            nn.Conv2d(220, 330, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),nn.BatchNorm2d(330))
        self.layer4 = nn.Sequential(
            nn.Conv2d(330, 175, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),nn.BatchNorm2d(175))
        self.layer5 = nn.Sequential(
            nn.Conv2d(175, 40, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),nn.BatchNorm2d(40))
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(40*5*5, 13)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.drop_out(out)
        out = self.layer3(out)
        out = self.drop_out(out)
        out = self.layer4(out)
        out = self.drop_out(out)
        out = self.layer5(out)
        # print(out.shape)
        out = out.reshape(out.size(0), -1)#flattening the network
        out = torch.sigmoid(self.fc1(out))
        return F.log_softmax(out, dim=1)
    
def train(args, model, device, train_loader, optimizer, epoch):#training the result
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        results = list(map(int, target))
        target = torch.tensor(results)#ensuring targe is in tensor format
        data, target = data.to(device), target.to(device)#putting
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            # out= open("record","a+")
            # out.write('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #     epoch, batch_idx * len(data), len(train_loader.dataset),
            #     100. * batch_idx / len(train_loader), loss.item()))

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_index ,(data, target) in enumerate(test_loader):#batch size is because it returns 2 variables
            results = list(map(int, target))
            target = torch.tensor(results)#ensuring targe is in tensor format
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    out= open("record","a+")
    # out.write('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        # test_loss, correct, len(test_loader.dataset),
        # 100. * correct / len(test_loader.dataset)))
def val(args, model, device, val_loader):
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_index ,(data, target) in enumerate(val_loader):#batch size is because it returns 2 variables
            results = list(map(int, target))
            target = torch.tensor(results)#ensuring targe is in tensor format
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    val_loss /= len(val_loader.dataset)

    print('\nVal set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        val_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))
    out= open("record","a+")
    out.write('\nVal set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        val_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))
class MyDataset(Dataset):
    def __init__(self, file_path,transform=None):
        with open(file_path, 'rb') as f:
            self.data = pickle.load(f)
        self.transform = transforms.Compose([
        transforms.ToPILImage(), # because the input dtype is numpy.ndarray
        transforms.RandomHorizontalFlip(0.5), # because this method is used for PIL Image dtype
        transforms.ToTensor(), # because inpus dtype is PIL Image
        ])

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        # make sure image is open the n standardize images as size 500,200
        label,image = self.data[index]
        image = Image.open(image)
        image = np.asarray(image)
        image = np.resize(image,(300,300)) 
        image = torch.tensor(image)
        
        
        if self.transform is not None:#if no specific transformation use preset 
            image = self.transform(image)
            
        return image, label

    
    class Compose(object):
        """Composes several transforms together.
        Args:
            transforms (list of ``Transform`` objects): list of transforms to compose.
        Example:
            >>> transforms.Compose([
            >>>     transforms.CenterCrop(10),
            >>>     transforms.ToTensor(),
            >>> ]) """

        def __init__(self, transforms):
            self.transforms = transforms

        def __call__(self, img):
            for t in self.transforms:
                img = t(img)
            return img

        def __repr__(self):
            format_string = self.__class__.__name__ + '('
            for t in self.transforms:
                format_string += '\n'
                format_string += '    {0}'.format(t)
            format_string += '\n)'
            return format_string

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch vids')
    parser.add_argument('--batch-size', type=int, default=30, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=40, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=4, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")


    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    transform = transforms.Compose([
    transforms.ToPILImage(), # because the input dtype is numpy.ndarray
    transforms.RandomHorizontalFlip(0.5), # because this method is used for PIL Image dtype
    transforms.ToTensor(), # because inpus dtype is PIL Image
    ])
    trainset= MyDataset('trainPCn.pkl',transform=transform)
    # train1 = trainset.__getitem__(1)
    testset = MyDataset('testPCn.pkl',transform=transform)
    valset = MyDataset('valPC.pkl',transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset,shuffle=True,batch_size=args.batch_size,num_workers=1)
    test_loader = torch.utils.data.DataLoader(testset,shuffle=True,batch_size=args.test_batch_size,num_workers=1)
    val_loader = torch.utils.data.DataLoader(valset,shuffle=True,batch_size=args.test_batch_size,num_workers=1)

    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        val(args, model, device, val_loader)
    test(args, model, device, test_loader)
    if (args.save_model):
        torch.save(model.state_dict(),"video.pt")
        
if __name__ == '__main__':
    main()
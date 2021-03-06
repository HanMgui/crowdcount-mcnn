import torch
import torch.nn as nn
from network import Conv2d, Block, Block2


class MCNN(nn.Module):
    '''
    Multi-column CNN 
        -Implementation of Single Image Crowd Counting via Multi-column CNN (Zhang et al.)
    '''
    
    def __init__(self, bn=False):
        super(MCNN, self).__init__()
        
        self.branch1 = nn.Sequential(Conv2d( 1, 16, 9, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Conv2d(16, 32, 7, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Conv2d(32, 16, 7, same_padding=True, bn=bn),
                                     Conv2d(16,  8, 7, same_padding=True, bn=bn))
        
        self.branch2 = nn.Sequential(Conv2d( 1, 20, 7, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Conv2d(20, 40, 5, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Conv2d(40, 20, 5, same_padding=True, bn=bn),
                                     Conv2d(20, 10, 5, same_padding=True, bn=bn))
        
        self.branch3 = nn.Sequential(Conv2d( 1, 24, 5, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Conv2d(24, 48, 3, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Conv2d(48, 24, 3, same_padding=True, bn=bn),
                                     Conv2d(24, 12, 3, same_padding=True, bn=bn))
        
        self.fuse = nn.Sequential(Conv2d( 30, 1, 1, same_padding=True, bn=bn,))
        
    def forward(self, im_data):
        x1 = self.branch1(im_data)
        x2 = self.branch2(im_data)
        x3 = self.branch3(im_data)
        x = torch.cat((x1,x2,x3),1)
        x = self.fuse(x)
        
        return x





class MCNNandMBv1(nn.Module):
    '''
    Multi-column CNN and MobileNetv1-version1
        -Implementation of Single Image Crowd Counting via Multi-column CNN (Zhang et al.)
    '''

    def __init__(self, bn=False):
        super(MCNNandMBv1, self).__init__()

        self.branch1 = nn.Sequential(Conv2d(1, 16, 9, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Block(16, 32, 7, same_padding=True, bn=bn),
                                     # Conv2d(16, 32, 7, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Block(32, 16, 7, same_padding=True, bn=bn),
                                     # Conv2d(32, 16, 7, same_padding=True, bn=bn),
                                     Block(16, 8, 7, same_padding=True, bn=bn))
                                     # Conv2d(16, 8, 7, same_padding=True, bn=bn))

        self.branch2 = nn.Sequential(Conv2d(1, 20, 7, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Block(20, 40, 5, same_padding=True, bn=bn),
                                     # Conv2d(20, 40, 5, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Block(40, 20, 5, same_padding=True, bn=bn),
                                     # Conv2d(40, 20, 5, same_padding=True, bn=bn),
                                     Block(20, 10, 5, same_padding=True, bn=bn))
                                     # Conv2d(20, 10, 5, same_padding=True, bn=bn))

        self.branch3 = nn.Sequential(Conv2d(1, 24, 5, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Block(24, 48, 3, same_padding=True, bn=bn),
                                     # Conv2d(24, 48, 3, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Block(48, 24, 3, same_padding=True, bn=bn),
                                     # Conv2d(48, 24, 3, same_padding=True, bn=bn),
                                     Block(24, 12, 3, same_padding=True, bn=bn))
                                     # Conv2d(24, 12, 3, same_padding=True, bn=bn))

        self.fuse = nn.Sequential(Conv2d(30, 1, 1, same_padding=True, bn=bn))

    def forward(self, im_data):
        x1 = self.branch1(im_data)
        x2 = self.branch2(im_data)
        x3 = self.branch3(im_data)
        x = torch.cat((x1, x2, x3), 1)
        x = self.fuse(x)

        return x


class MCNNandMBv1_1(nn.Module):
    '''
    Multi-column CNN and MobileNetv1-version1
        -Implementation of Single Image Crowd Counting via Multi-column CNN (Zhang et al.)
    '''

    def __init__(self, bn=False):
        super(MCNNandMBv1_1, self).__init__()

        self.branch1 = nn.Sequential(Conv2d(1, 16, 9, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Block(16, 32, 3, same_padding=True, bn=bn),
                                     Block(32, 32, 3, same_padding=True, bn=bn),
                                     Block(32, 32, 3, same_padding=True, bn=bn),
                                     # Conv2d(16, 32, 7, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Block(32, 16, 3, same_padding=True, bn=bn),
                                     Block(16, 16, 3, same_padding=True, bn=bn),
                                     Block(16, 16, 3, same_padding=True, bn=bn),
                                     # Conv2d(32, 16, 7, same_padding=True, bn=bn),
                                     Block(16, 8, 3, same_padding=True, bn=bn),
                                     Block(8, 8, 3, same_padding=True, bn=bn),
                                     Block(8, 8, 3, same_padding=True, bn=bn))
                                     # Conv2d(16, 8, 7, same_padding=True, bn=bn))

        self.branch2 = nn.Sequential(Conv2d(1, 20, 7, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Block(20, 40, 3, same_padding=True, bn=bn),
                                     Block(40, 40, 3, same_padding=True, bn=bn),
                                     # Conv2d(20, 40, 5, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Block(40, 20, 3, same_padding=True, bn=bn),
                                     Block(20, 20, 3, same_padding=True, bn=bn),
                                     # Conv2d(40, 20, 5, same_padding=True, bn=bn),
                                     Block(20, 10, 3, same_padding=True, bn=bn),
                                     Block(10, 10, 3, same_padding=True, bn=bn))
                                     # Conv2d(20, 10, 5, same_padding=True, bn=bn))

        self.branch3 = nn.Sequential(Conv2d(1, 24, 5, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Block(24, 48, 3, same_padding=True, bn=bn),
                                     # Conv2d(24, 48, 3, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Block(48, 24, 3, same_padding=True, bn=bn),
                                     # Conv2d(48, 24, 3, same_padding=True, bn=bn),
                                     Block(24, 12, 3, same_padding=True, bn=bn))
                                     # Conv2d(24, 12, 3, same_padding=True, bn=bn))

        self.fuse = nn.Sequential(Conv2d(30, 1, 1, same_padding=True, bn=bn))

    def forward(self, im_data):
        x1 = self.branch1(im_data)
        x2 = self.branch2(im_data)
        x3 = self.branch3(im_data)
        x = torch.cat((x1, x2, x3), 1)
        x = self.fuse(x)

        return x

class MCNNandMBv1_2(nn.Module):
    '''
    Multi-column CNN and MobileNetv1-version1
        -Implementation of Single Image Crowd Counting via Multi-column CNN (Zhang et al.)
    '''

    def __init__(self, bn=False):
        super(MCNNandMBv1_2, self).__init__()

        self.branch1 = nn.Sequential(Conv2d(1, 16, 9, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Block2(16, 32, 7, same_padding=True, bn=bn),
                                     # Conv2d(16, 32, 7, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Block2(32, 16, 7, same_padding=True, bn=bn),
                                     # Conv2d(32, 16, 7, same_padding=True, bn=bn),
                                     Block2(16, 8, 7, same_padding=True, bn=bn))
                                     # Conv2d(16, 8, 7, same_padding=True, bn=bn))

        self.branch2 = nn.Sequential(Conv2d(1, 20, 7, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Block2(20, 40, 5, same_padding=True, bn=bn),
                                     # Conv2d(20, 40, 5, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Block2(40, 20, 5, same_padding=True, bn=bn),
                                     # Conv2d(40, 20, 5, same_padding=True, bn=bn),
                                     Block2(20, 10, 5, same_padding=True, bn=bn))
                                     # Conv2d(20, 10, 5, same_padding=True, bn=bn))

        self.branch3 = nn.Sequential(Conv2d(1, 24, 5, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Block2(24, 48, 3, same_padding=True, bn=bn),
                                     # Conv2d(24, 48, 3, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Block2(48, 24, 3, same_padding=True, bn=bn),
                                     # Conv2d(48, 24, 3, same_padding=True, bn=bn),
                                     Block2(24, 12, 3, same_padding=True, bn=bn))
                                     # Conv2d(24, 12, 3, same_padding=True, bn=bn))

        self.fuse = nn.Sequential(Conv2d(30, 1, 1, same_padding=True, bn=bn))

    def forward(self, im_data):
        x1 = self.branch1(im_data)
        x2 = self.branch2(im_data)
        x3 = self.branch3(im_data)
        x = torch.cat((x1, x2, x3), 1)
        x = self.fuse(x)

        return x
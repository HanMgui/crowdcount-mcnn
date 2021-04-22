import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, relu=True, same_padding=False, bn=False):
        super(Conv2d, self).__init__()
        padding = int((kernel_size - 1) / 2) if same_padding else 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class FC(nn.Module):
    def __init__(self, in_features, out_features, relu=True):
        super(FC, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.fc(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Block(nn.Module):
    '''Depthwise conv + Pointwise conv'''
    def __init__(self, in_planes, out_planes, kernel_size=3,stride=1,same_padding=True,bn=False):
        super(Block, self).__init__()
        padding = int((kernel_size - 1) / 2) if same_padding else 0
        # Depthwise 卷积，3*3 的卷积核，分为 in_planes，即各层单独进行卷积
        # 输入为in_planes,输出也为in_planes
        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_planes, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        # Pointwise 卷积，1*1 的卷积核
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes, eps=0.001, momentum=0, affine=True) if bn else None
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        if self.bn2 == None:
            out = self.relu2(self.conv2(out))
        else:
            out = self.relu2(self.bn2(self.conv2(out)))
        return out

class Block2(nn.Module):
    '''Depthwise conv + Pointwise conv'''
    def __init__(self, in_planes, out_planes, kernel_size=3,stride=1,same_padding=True,bn=False):
        super(Block2, self).__init__()
        padding = 1 if same_padding else 0
        # Depthwise 卷积，3*3 的卷积核，分为 in_planes，即各层单独进行卷积
        # 输入为in_planes,输出也为in_planes
        for _ in range(0, int(kernel_size / 2)):
            self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride, padding=padding,
                                   groups=in_planes, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        # Pointwise 卷积，1*1 的卷积核
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes, eps=0.001, momentum=0, affine=True) if bn else None
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        if self.bn2 == None:
            out = self.relu2(self.conv2(out))
        else:
            out = self.relu2(self.bn2(self.conv2(out)))
        return out

def save_net(fname, net):
    import h5py
    h5f = h5py.File(fname, mode='w')
    for k, v in net.state_dict().items():
        h5f.create_dataset(k, data=v.cpu().numpy())


def load_net(fname, net):
    import h5py
    h5f = h5py.File(fname, mode='r')
    for k, v in net.state_dict().items():        
        param = torch.from_numpy(np.asarray(h5f[k]))         
        v.copy_(param)


def np_to_variable(x, is_cuda=True, is_training=False, dtype=torch.FloatTensor):
    if is_training:
        v = Variable(torch.from_numpy(x).type(dtype))
    else:
        with torch.no_grad():
            v = Variable(torch.from_numpy(x).type(dtype))
    if is_cuda:
        v = v.cuda()
    return v


def set_trainable(model, requires_grad):
    for param in model.parameters():
        param.requires_grad = requires_grad


def weights_normal_init(model, dev=0.01):
    if isinstance(model, list):
        for m in model:
            weights_normal_init(m, dev)
    else:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):                
                #print torch.sum(m.weight)
                m.weight.data.normal_(0.0, dev)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, dev)

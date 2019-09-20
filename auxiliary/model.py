from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

from utils import *
from dataset_TC import *


#TANGENT CONVOLUTION ENCODER
class TangentConv(nn.Module):
    def __init__(self, num_points = 2500, input_channels = 4, filter_size = 9):
        super(TangentConv, self).__init__()
        self.num_points = num_points

        self.conv_11 = torch.nn.Conv2d(input_channels, 32, (1, 9))
        self.conv_12 = torch.nn.Conv2d(32, 32, (1, 9))
        self.conv_21 = torch.nn.Conv2d(33, 64, (1, 9))
        self.conv_22 = torch.nn.Conv2d(64, 64, (1, 9))
        self.conv_latent = torch.nn.Conv2d(64, 1, (1, 1))

    def forward(self, x, masks):
        '''
        Tangent Convolution Encoder network.
        :params: x     : normals from PointCloud object (numpy array batch_sizex2500x3)
                 masks : Convolution and Pooling indices as well as Pooling mask and
                            depth layer, all from Tangent Convolutions precomputing
                            step (numpy arrays)
        :return x : Encoded pointcloud representation
        '''

        x = torch.cat((masks[0][0], x), dim=2)
        # x = masks[0][0]

        ### Tangent Convolution Layer 1
        x = self.tangconv(x, masks[0][1], masks[0][3])
        x = F.leaky_relu(self.conv_11(x), negative_slope=0.1)
        x = torch.transpose(x, 1, 2).squeeze()

        ### Tangent Convolution Layer 2
        x = self.tangconv(x, masks[0][1])
        x = F.leaky_relu(self.conv_12(x), negative_slope=0.1)
        x = torch.transpose(x, 1, 2).squeeze()

        x = self.tangconv(x, masks[0][1])
        x = F.leaky_relu(self.conv_12(x), negative_slope=0.1)
        x = torch.transpose(x, 1, 2).squeeze()

        ### Tangent Pooling Layer 1
        x = self.tangpool(x, masks[1][2], masks[1][4])

        ### Tangent Convolution Layer 3
        x = self.tangconv(x, masks[1][1], masks[1][3])
        x = F.leaky_relu(self.conv_21(x), negative_slope=0.1)
        x = torch.transpose(x, 1, 2).squeeze()

        ### Tangent Convolution Layer 4
        x = self.tangconv(x, masks[1][1])
        x = F.leaky_relu(self.conv_22(x), negative_slope=0.1)
        x = torch.transpose(x, 1, 2).squeeze()

        x = self.tangconv(x, masks[1][1])
        x = F.leaky_relu(self.conv_22(x), negative_slope=0.1)
        x = torch.transpose(x, 1, 2).squeeze()

        ### Bring to latent representation (num_points/4, 1) size
        x = torch.transpose(torch.unsqueeze(x, 1), 1, 3)
        x = F.leaky_relu(self.conv_latent(x), negative_slope=0.1)
        x = torch.transpose(x.squeeze().unsqueeze(2), 1, 2)

        return x


    def tangconv(self, x, conv_ind, depth=None):
        conv_ind = torch.cuda.LongTensor(conv_ind.long())

        M = x[np.arange(0, conv_ind.shape[0])[:, None, None], conv_ind]
        M = torch.cuda.FloatTensor(M.float())

        if depth is not None:
            depth = torch.cuda.FloatTensor(depth.float())
            x = torch.cat((M, torch.unsqueeze(depth, 3)), dim=3)
        else:
            x = M
            
        x = torch.transpose(torch.transpose(x, 2, 3), 1, 2)

        return x

    def tangpool(self, x, pool_ind, pool_mask):
        zeros = torch.zeros([x.shape[0], 1, x.shape[2]]).cuda()
        x = torch.cat((x, zeros), dim=1)
        pool_input = x[np.arange(0, pool_ind.shape[0])[:, None, None], pool_ind]

        x = torch.sum(pool_input, dim=2)

        return x


# ATLASNET DECODER
class PointGenCon(nn.Module):
    def __init__(self, bottleneck_size = 2500):
        self.bottleneck_size = bottleneck_size
        super(PointGenCon, self).__init__()
        self.conv1 = torch.nn.Conv1d(self.bottleneck_size, self.bottleneck_size, 1)
        self.conv2 = torch.nn.Conv1d(self.bottleneck_size, self.bottleneck_size//2, 1)
        self.conv22 = torch.nn.Conv1d(self.bottleneck_size//2, self.bottleneck_size//2, 1)
        self.conv3 = torch.nn.Conv1d(self.bottleneck_size//2, self.bottleneck_size//4, 1)
        self.conv32 = torch.nn.Conv1d(self.bottleneck_size//4, self.bottleneck_size//4, 1)
        self.conv4 = torch.nn.Conv1d(self.bottleneck_size//4, 3, 1)

        self.th = nn.Tanh()
        self.bn1 = torch.nn.BatchNorm1d(self.bottleneck_size)
        self.bn2 = torch.nn.BatchNorm1d(self.bottleneck_size//2)
        self.bn3 = torch.nn.BatchNorm1d(self.bottleneck_size//4)

    def forward(self, x):
        batchsize = x.size()[0]
        # print(x.size())
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.conv22(x))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.conv32(x))
        x = self.th(self.conv4(x))
        return x


class TangConv_AtlasNet(nn.Module):
    def __init__(self, num_points = 2500, bottleneck_size = 1024,
                 nb_primitives = 1):
        super(TangConv_AtlasNet, self).__init__()
        self.num_points = num_points
        self.bottleneck_size = bottleneck_size
        self.nb_primitives = nb_primitives

        self.encoder = TangentConv(num_points=2500, input_channels=7, filter_size=9)
        self.encoderlin = nn.Linear(self.num_points//2, self.bottleneck_size)
        self.encoderbn = nn.BatchNorm1d(self.bottleneck_size)
        self.decoder = nn.ModuleList([PointGenCon(bottleneck_size=2+self.bottleneck_size)
                                      for i in range(0, self.nb_primitives)])


    def forward(self, x, masks):
        x = self.encoder(x, masks)
        x = self.encoderlin(x)
        x = F.relu(self.encoderbn(torch.transpose(x, 1, 2)))

        outs = []
        for i in range(0,self.nb_primitives):
            rand_grid = Variable(torch.cuda.FloatTensor(x.size(0),2,self.num_points//self.nb_primitives))
            rand_grid.data.uniform_(0,1)
            y = x.expand(x.size(0),x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat( (rand_grid, y), 1).contiguous()
            outs.append(self.decoder[i](y))
        return torch.cat(outs,2).contiguous().transpose(2,1).contiguous()

    def forward_inference(self, x, grid):
        x = self.encoder(x, masks)
        outs = []
        for i in range(0,self.nb_primitives):
            rand_grid = Variable(torch.cuda.FloatTensor(grid[i]))
            rand_grid = rand_grid.transpose(0,1).contiguous().unsqueeze(0)
            rand_grid = rand_grid.expand(x.size(0),rand_grid.size(1), rand_grid.size(2)).contiguous()
            y = x.unsqueeze(2).expand(x.size(0),x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat( (rand_grid, y), 1).contiguous()
            outs.append(self.decoder[i](y))
        return torch.cat(outs,2).contiguous().transpose(2,1).contiguous()


if __name__ == '__main__':
    # For testing purposes
    print('Testing TangConv...')
    d  =  TangConvShapeNet(class_choice=None, balanced=False, train=True, npoints=2500)

    masks, normals, _, _ = d.__getitem__(50)
    network = TangConv_AtlasNet(num_points=2500, bottleneck_size=1024, nb_primitives=1)
    network.cuda()
    network.apply(weights_init)
    pointsReconstructed  = network(normals, masks)

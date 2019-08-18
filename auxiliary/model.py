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

utils_path = '/home/parker/code/AtlasNet/utils/'
sys.path.append(utils_path)

from cloud import get_pooling_mask

#UTILITIES
class STN3d(nn.Module):
    def __init__(self, num_points = 2500):
        super(STN3d, self).__init__()
        self.num_points = num_points
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x,_ = torch.max(x, 2)
        x = x.view(-1, 1024)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1,9).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class PointNetfeat(nn.Module):
    def __init__(self, num_points = 2500, global_feat = True, trans = False):
        super(PointNetfeat, self).__init__()
        self.stn = STN3d(num_points = num_points)
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)

        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(1024)
        self.trans = trans


        #self.mp1 = torch.nn.MaxPool1d(num_points)
        self.num_points = num_points
        self.global_feat = global_feat

    def forward(self, x):
        batchsize = x.size()[0]
        if self.trans:
            trans = self.stn(x)
            x = x.transpose(2,1)
            x = torch.bmm(x, trans)
            x = x.transpose(2,1)
        x = F.relu(self.bn1(self.conv1(x)))
        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x,_ = torch.max(x, 2)
        x = x.view(-1, 1024)
        if self.trans:
            if self.global_feat:
                return x, trans
            else:
                x = x.view(-1, 1024, 1).repeat(1, 1, self.num_points)
                return torch.cat([x, pointfeat], 1), trans
        else:
            return x


class PointNetfeatNormal(nn.Module):
    def __init__(self, num_points = 2500, global_feat = True, trans = False):
        super(PointNetfeatNormal, self).__init__()
        self.stn = STN3d(num_points = num_points)
        self.conv1 = torch.nn.Conv1d(6, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)

        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(1024)
        self.trans = trans


        #self.mp1 = torch.nn.MaxPool1d(num_points)
        self.num_points = num_points
        self.global_feat = global_feat
    def forward(self, x):
        batchsize = x.size()[0]
        if self.trans:
            trans = self.stn(x)
            x = x.transpose(2,1)
            x = torch.bmm(x, trans)
            x = x.transpose(2,1)
        x = F.relu(self.bn1(self.conv1(x)))
        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x,_ = torch.max(x, 2)
        x = x.view(-1, 1024)
        if self.trans:
            if self.global_feat:
                return x, trans
            else:
                x = x.view(-1, 1024, 1).repeat(1, 1, self.num_points)
                return torch.cat([x, pointfeat], 1), trans
        else:
            return x


class TangentConv(nn.Module):
    # def __init__(self, masks, num_points = 2500, input_channels = 4, filter_size = 9):
    def __init__(self, num_points = 2500, input_channels = 4, filter_size = 9):
        super(TangentConv, self).__init__()
        self.num_points = num_points
        # self.masks = masks

        self.conv_11 = torch.nn.Conv2d(input_channels, 32, (1, 9))
        self.conv_12 = torch.nn.Conv2d(32, 32, (1, 9))
        self.conv_21 = torch.nn.Conv2d(33, 64, (1, 9))
        self.conv_22 = torch.nn.Conv2d(64, 64, (1, 9))
        self.conv_31 = torch.nn.Conv2d(65, 128, (1, 9))
        self.conv_latent = torch.nn.Conv2d(128, 1, (1, 1))

    def forward(self, x, masks):
        '''
        Tangent Convolution Encoder network.
        :params: x     : normals from PointCloud object (numpy array 2500x3)
                 masks : Convolution and Pooling indices as well as Pooling mask and
                            depth layer, all from Tangent Convolutions precomputing
                            step (numpy arrays)
        :return x : Encoded pointcloud representation
        '''

        ### Tangent Convolution Layer 1
        x = self.tangconv(x, masks[0][1], masks[0][3])
        x = F.leaky_relu(self.conv_11(x.unsqueeze(0)), negative_slope=0.1)
        x = torch.transpose(x, 1, 2).squeeze()
        # print(x.shape)
        # input('TANGCONV FIRST LAYER')

        ### Tangent Convolution Layer 2
        x = self.tangconv(x, masks[0][1])
        x = F.leaky_relu(self.conv_12(x.unsqueeze(0)), negative_slope=0.1)
        x = torch.transpose(x, 1, 2).squeeze()
        # print(x.shape)
        # input('TANGCONV SECOND LAYER')

        ### Tangent Pooling Layer 1
        x = self.tangpool(x, masks[1][2], masks[1][4])
        # print(x.shape)
        # input('TANGCONV FIRST POOL')

        ### Tangent Convolution Layer 3
        x = self.tangconv(x, masks[1][1], masks[1][3])
        x = F.leaky_relu(self.conv_21(x.unsqueeze(0)), negative_slope=0.1)
        x = torch.transpose(x, 1, 2).squeeze()
        # print(x.shape)
        # input('TANGCONV THIRD LAYER')

        ### Tangent Convolution Layer 4
        x = self.tangconv(x, masks[1][1])
        x = F.leaky_relu(self.conv_22(x.unsqueeze(0)), negative_slope=0.1)
        x = torch.transpose(x, 1, 2).squeeze()
        # print(x.shape)
        # input('TANGCONV FOURTH LAYER')

        ### Tangent Pooling Layer 2
        # Maybe unsqueeze first
        x = self.tangpool(x, masks[2][2], masks[2][4])
        # print(x.shape)
        # input('TANGCONV SECOND POOL')

        ### Tangent Convolution Layer 5
        x = self.tangconv(x, masks[2][1], masks[2][3])
        x = F.leaky_relu(self.conv_31(x.unsqueeze(0)), negative_slope=0.1)
        x = torch.transpose(x, 1, 2).squeeze()
        # print(x.shape)
        # input('TANGCONV FIFTH LAYER')

        ### Bring to latent representation (num_points/4, 1) size
        x = self.latentconv(x)
        x = F.leaky_relu(self.conv_latent(x.unsqueeze(0)), negative_slope=0.1)
        x = x.squeeze()
        x = torch.transpose(x.unsqueeze(1), 0, 1)
        # print(x.shape)
        # input('LATENT LAYER')

        return x


    def tangconv(self, x, conv_ind, depth=None):
        conv_ind = torch.cuda.LongTensor(conv_ind.long())

        M = torch.empty(conv_ind.shape[0], conv_ind.shape[1], x.shape[1]).cuda()
        print(x.size())
        print(M.size())
        input()

        for i in range(0, M.shape[0]):
            for j in range(0, M.shape[1]):
                M[i, j, :] = x[conv_ind[i, j], :]

        if depth is not None:
            depth = torch.cuda.FloatTensor(depth)
            x = torch.cat((M, torch.unsqueeze(depth, 2)), dim=2)
        else:
            x = M

        x = torch.transpose(torch.transpose(x, 0, 1), 0, 2)

        return x

    def tangpool(self, x, pool_ind, pool_mask):
        pool_input = torch.empty(pool_ind.shape[0], pool_ind.shape[1], x.shape[1]).cuda()
        for i in range(0, pool_input.shape[0]):
            for j in range(0, pool_input.shape[1]):
                if pool_ind[i, j] == -1:
                    pool_input[i, j, :] = 0
                else:
                    pool_input[i, j, :] = x[int(pool_ind[i, j]), :]

        # broadcast_input = torch.unsqueeze(torch.from_numpy(pool_mask).cuda(), 2)
        # broadcast_mask = broadcast_input.repeat(1, 1, pool_input.shape[2])

        # x = torch.sum(x * broadcast_mask, dim=1)
        x = torch.sum(pool_input, dim=1)

        return x

    def latentconv(self, x):
        x = torch.transpose(torch.unsqueeze(x, 2), 0, 1)

        return x


#OUR METHOD
import resnet

class PointGenCon(nn.Module):
    def __init__(self, bottleneck_size = 2500):
        self.bottleneck_size = bottleneck_size
        super(PointGenCon, self).__init__()
        self.conv1 = torch.nn.Conv1d(self.bottleneck_size, self.bottleneck_size, 1)
        self.conv2 = torch.nn.Conv1d(self.bottleneck_size, self.bottleneck_size//2, 1)
        self.conv3 = torch.nn.Conv1d(self.bottleneck_size//2, self.bottleneck_size//4, 1)
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
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.th(self.conv4(x))
        return x


class SVR_AtlasNet(nn.Module):
    def __init__(self, num_points = 2048, bottleneck_size = 1024, nb_primitives = 5, pretrained_encoder = False, cuda=True):
        super(SVR_AtlasNet, self).__init__()
        self.usecuda = cuda
        self.num_points = num_points
        self.bottleneck_size = bottleneck_size
        self.nb_primitives = nb_primitives
        self.pretrained_encoder = pretrained_encoder
        self.encoder = resnet.resnet18(pretrained=self.pretrained_encoder, num_classes=1024)
        self.decoder = nn.ModuleList([PointGenCon(bottleneck_size = 2 +self.bottleneck_size) for i in range(0, self.nb_primitives)])

    def forward(self, x):
        x = x[:,:3,:,:].contiguous()
        x = self.encoder(x)
        outs = []
        for i in range(0, self.nb_primitives):
            rand_grid = Variable(torch.cuda.FloatTensor(x.size(0), 2, self.num_points//self.nb_primitives))
            rand_grid.data.uniform_(0, 1)
            y = x.unsqueeze(2).expand(x.size(0), x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat( (rand_grid, y.type_as(rand_grid)), 1).contiguous()
            outs.append(self.decoder[i](y))
        return torch.cat(outs, 2).contiguous().transpose(2,1).contiguous()

    def decode(self, x):
        outs = []
        for i in range(0, self.nb_primitives):
            rand_grid = Variable(torch.cuda.FloatTensor(x.size(0), 2, self.num_points//self.nb_primitives))
            rand_grid.data.uniform_(0, 1)
            y = x.unsqueeze(2).expand(x.size(0), x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat( (rand_grid, y.type_as(rand_grid)), 1).contiguous()
            outs.append(self.decoder[i](y))
        return torch.cat(outs, 2).contiguous().transpose(2,1).contiguous()

    def forward_inference(self, x, grid):
        x = self.encoder(x)
        outs = []
        for i in range(0, self.nb_primitives):
            if self.usecuda:
                rand_grid = Variable(torch.cuda.FloatTensor(grid[i]))
            else:
                rand_grid = Variable(torch.FloatTensor(grid[i]))

            rand_grid = rand_grid.transpose(0, 1).contiguous().unsqueeze(0)
            rand_grid = rand_grid.expand(x.size(0), rand_grid.size(1), rand_grid.size(2)).contiguous()
            # print(rand_grid.sizerand_grid())
            y = x.unsqueeze(2).expand(x.size(0), x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat( (rand_grid, y), 1).contiguous()
            outs.append(self.decoder[i](y))
        return torch.cat(outs, 2).contiguous().transpose(2,1).contiguous()

    def forward_inference_from_latent_space(self, x, grid):
        outs = []
        for i in range(0, self.nb_primitives):
            rand_grid = Variable(torch.cuda.FloatTensor(grid[i]))
            rand_grid = rand_grid.transpose(0, 1).contiguous().unsqueeze(0)
            rand_grid = rand_grid.expand(x.size(0), rand_grid.size(1), rand_grid.size(2)).contiguous()
            # print(rand_grid.sizerand_grid())
            y = x.unsqueeze(2).expand(x.size(0), x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat( (rand_grid, y), 1).contiguous()
            outs.append(self.decoder[i](y))
        return torch.cat(outs, 2).contiguous().transpose(2,1).contiguous()


class AE_AtlasNet(nn.Module):
    def __init__(self, num_points = 2048, bottleneck_size = 1024, nb_primitives = 1):
        super(AE_AtlasNet, self).__init__()
        self.num_points = num_points
        self.bottleneck_size = bottleneck_size
        self.nb_primitives = nb_primitives
        self.encoder = nn.Sequential(
        PointNetfeat(num_points, global_feat=True, trans = False),
        nn.Linear(1024, self.bottleneck_size),
        nn.BatchNorm1d(self.bottleneck_size),
        nn.ReLU()
        )
        self.decoder = nn.ModuleList([PointGenCon(bottleneck_size = 2 +self.bottleneck_size) for i in range(0,self.nb_primitives)])


    def forward(self, x):
        x = self.encoder(x)
        outs = []
        for i in range(0,self.nb_primitives):
            rand_grid = Variable(torch.cuda.FloatTensor(x.size(0),2,self.num_points//self.nb_primitives))
            rand_grid.data.uniform_(0,1)
            y = x.unsqueeze(2).expand(x.size(0),x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat( (rand_grid, y), 1).contiguous()
            outs.append(self.decoder[i](y))
        return torch.cat(outs,2).contiguous().transpose(2,1).contiguous()

    def forward_inference(self, x, grid):
        x = self.encoder(x)
        outs = []
        for i in range(0,self.nb_primitives):
            rand_grid = Variable(torch.cuda.FloatTensor(grid[i]))
            rand_grid = rand_grid.transpose(0,1).contiguous().unsqueeze(0)
            rand_grid = rand_grid.expand(x.size(0),rand_grid.size(1), rand_grid.size(2)).contiguous()
            # print(rand_grid.sizerand_grid())
            y = x.unsqueeze(2).expand(x.size(0),x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat( (rand_grid, y), 1).contiguous()
            outs.append(self.decoder[i](y))
        return torch.cat(outs,2).contiguous().transpose(2,1).contiguous()

    def forward_inference_from_latent_space(self, x, grid):
        outs = []
        for i in range(0,self.nb_primitives):
            rand_grid = Variable(torch.cuda.FloatTensor(grid[i]))
            rand_grid = rand_grid.transpose(0,1).contiguous().unsqueeze(0)
            rand_grid = rand_grid.expand(x.size(0),rand_grid.size(1), rand_grid.size(2)).contiguous()
            # print(rand_grid.sizerand_grid())
            y = x.unsqueeze(2).expand(x.size(0),x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat( (rand_grid, y), 1).contiguous()
            outs.append(self.decoder[i](y))
        return torch.cat(outs,2).contiguous().transpose(2,1).contiguous()


class TangConv_AtlasNet(nn.Module):
    # def __init__(self, masks, num_points = 2500, bottleneck_size = 1024,
    #              nb_primitives = 1):
    def __init__(self, num_points = 2500, bottleneck_size = 1024,
                 nb_primitives = 1):
        super(TangConv_AtlasNet, self).__init__()
        # self.masks = mask
        self.num_points = num_points
        self.bottleneck_size = bottleneck_size
        self.nb_primitives = nb_primitives
        # self.encoder = nn.Sequential(TangentConv(num_points=2500, input_channels=4, filter_size=9),
        #                               nn.Linear(self.num_points//4, self.bottleneck_size),
        #                               nn.BatchNorm1d(self.bottleneck_size),
        #                               nn.ReLU()
        #                             )
        self.encoder = TangentConv(num_points=2500, input_channels=4, filter_size=9)
        self.postencoder = nn.Sequential(nn.Linear(self.num_points//4, self.bottleneck_size),
                                         nn.BatchNorm1d(self.bottleneck_size),
                                         nn.ReLU()
                                        )
        self.decoder = nn.ModuleList([PointGenCon(bottleneck_size=2+self.bottleneck_size)
                                      for i in range(0, self.nb_primitives)])


    def forward(self, x, masks):
        x = self.encoder(x, masks)
        x = self.postencoder(x)
        outs = []
        for i in range(0,self.nb_primitives):
            rand_grid = Variable(torch.cuda.FloatTensor(x.size(0),2,self.num_points//self.nb_primitives))
            rand_grid.data.uniform_(0,1)
            y = x.unsqueeze(2).expand(x.size(0),x.size(1), rand_grid.size(2)).contiguous()
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
            # print(rand_grid.sizerand_grid())
            y = x.unsqueeze(2).expand(x.size(0),x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat( (rand_grid, y), 1).contiguous()
            outs.append(self.decoder[i](y))
        return torch.cat(outs,2).contiguous().transpose(2,1).contiguous()

    def forward_inference_from_latent_space(self, x, grid):
        outs = []
        for i in range(0,self.nb_primitives):
            rand_grid = Variable(torch.cuda.FloatTensor(grid[i]))
            rand_grid = rand_grid.transpose(0,1).contiguous().unsqueeze(0)
            rand_grid = rand_grid.expand(x.size(0),rand_grid.size(1), rand_grid.size(2)).contiguous()
            # print(rand_grid.sizerand_grid())
            y = x.unsqueeze(2).expand(x.size(0),x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat( (rand_grid, y), 1).contiguous()
            outs.append(self.decoder[i](y))
        return torch.cat(outs,2).contiguous().transpose(2,1).contiguous()


#TEST with spheric noise
class AE_AtlasNet_SPHERE(nn.Module):
    def __init__(self, num_points = 2048, bottleneck_size = 1024, nb_primitives = 1):
        super(AE_AtlasNet_SPHERE, self).__init__()
        self.num_points = num_points
        self.bottleneck_size = bottleneck_size
        self.nb_primitives = nb_primitives
        self.encoder = nn.Sequential(
        PointNetfeat(num_points, global_feat=True, trans = False),
        nn.Linear(1024, self.bottleneck_size),
        nn.BatchNorm1d(self.bottleneck_size),
        nn.ReLU()
        )
        self.decoder = nn.ModuleList([PointGenCon(bottleneck_size = 3 +self.bottleneck_size) for i in range(0,self.nb_primitives)])


    def forward(self, x):
        x = self.encoder(x)
        outs = []
        for i in range(0,self.nb_primitives):
            rand_grid = Variable(torch.cuda.FloatTensor(x.size(0),3,self.num_points//self.nb_primitives)) #sample points randomly
            rand_grid.data.normal_(0,1)
            rand_grid = rand_grid / torch.sqrt(torch.sum(rand_grid**2, dim=1, keepdim=True)).expand(x.size(0),3,self.num_points//self.nb_primitives)
            #assert a number of things like norm/visdom... then copy to other functions
            y = x.unsqueeze(2).expand(x.size(0),x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat( (rand_grid, y), 1).contiguous()
            outs.append(self.decoder[i](y))
        return torch.cat(outs,2).contiguous().transpose(2,1).contiguous()

    def forward_inference(self, x, grid):
        x = self.encoder(x)
        outs = []
        for i in range(0,self.nb_primitives):
            grid = grid.contiguous().unsqueeze(0)
            grid = Variable(grid.expand(x.size(0),grid.size(1), grid.size(2)).contiguous())
            # print(grid.sizegrid())
            y = x.unsqueeze(2).expand(x.size(0),x.size(1), grid.size(2)).contiguous()
            y = torch.cat( (grid, y), 1).contiguous()
            outs.append(self.decoder[i](y))
        return torch.cat(outs,2).contiguous().transpose(2,1).contiguous()

    def forward_inference_from_latent_space(self, x, grid):
        outs = []
        for i in range(0,self.nb_primitives):
            grid = grid.contiguous().unsqueeze(0)
            grid = grid.expand(x.size(0),grid.size(1), grid.size(2)).contiguous()
            # print(grid.sizegrid())
            y = x.unsqueeze(2).expand(x.size(0),x.size(1), grid.size(2)).contiguous()
            y = torch.cat( (grid, y), 1).contiguous()
            outs.append(self.decoder[i](y))
        return torch.cat(outs,2).contiguous().transpose(2,1).contiguous()


class SVR_AtlasNet_SPHERE(nn.Module):
    def __init__(self, num_points = 2048, bottleneck_size = 1024, nb_primitives = 1, pretrained_encoder=False):
        super(SVR_AtlasNet_SPHERE, self).__init__()
        self.num_points = num_points
        self.pretrained_encoder = pretrained_encoder

        self.bottleneck_size = bottleneck_size
        self.nb_primitives = nb_primitives
        self.encoder = resnet.resnet18(pretrained=self.pretrained_encoder, num_classes=1024)
        self.decoder = nn.ModuleList([PointGenCon(bottleneck_size = 3 +self.bottleneck_size) for i in range(0,self.nb_primitives)])


    def forward(self, x):
        x = x[:,:3,:,:].contiguous()
        x = self.encoder(x)
        outs = []
        for i in range(0,self.nb_primitives):
            rand_grid = Variable(torch.cuda.FloatTensor(x.size(0),3,self.num_points//self.nb_primitives)) #sample points randomly
            rand_grid.data.normal_(0,1)
            rand_grid = rand_grid / torch.sqrt(torch.sum(rand_grid**2, dim=1, keepdim=True)).expand(x.size(0),3,self.num_points//self.nb_primitives)
            #assert a number of things like norm/visdom... then copy to other functions
            y = x.unsqueeze(2).expand(x.size(0),x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat( (rand_grid, y), 1).contiguous()
            outs.append(self.decoder[i](y))
        return torch.cat(outs,2).contiguous().transpose(2,1).contiguous()

    def forward_inference(self, x, grid):
        x = x[:,:3,:,:].contiguous()

        x = self.encoder(x)
        outs = []
        for i in range(0,self.nb_primitives):
            grid = grid.contiguous().unsqueeze(0)
            grid = Variable(grid.expand(x.size(0),grid.size(1), grid.size(2)).contiguous())
            # print(grid.sizegrid())
            y = x.unsqueeze(2).expand(x.size(0),x.size(1), grid.size(2)).contiguous()
            y = torch.cat( (grid, y), 1).contiguous()
            outs.append(self.decoder[i](y))
        return torch.cat(outs,2).contiguous().transpose(2,1).contiguous()

    def forward_inference_from_latent_space(self, x, grid):
        outs = []
        for i in range(0,self.nb_primitives):
            grid = grid.contiguous().unsqueeze(0)
            grid = grid.expand(x.size(0),grid.size(1), grid.size(2)).contiguous()
            # print(grid.sizegrid())
            y = x.unsqueeze(2).expand(x.size(0),x.size(1), grid.size(2)).contiguous()
            y = torch.cat( (grid, y), 1).contiguous()
            outs.append(self.decoder[i](y))
        return torch.cat(outs,2).contiguous().transpose(2,1).contiguous()


#BASELINE
class PointDecoder(nn.Module):
    def __init__(self, num_points = 2048, bottleneck_size = 1024):
        super(PointDecoder, self).__init__()
        self.num_points = num_points
        self.bottleneck_size = bottleneck_size
        self.bn1 = torch.nn.BatchNorm1d(bottleneck_size)
        self.bn2 = torch.nn.BatchNorm1d(bottleneck_size//2)
        self.bn3 = torch.nn.BatchNorm1d(bottleneck_size//4)
        self.fc1 = nn.Linear(self.bottleneck_size, bottleneck_size)
        self.fc2 = nn.Linear(self.bottleneck_size, bottleneck_size//2)
        self.fc3 = nn.Linear(bottleneck_size//2, bottleneck_size//4)
        self.fc4 = nn.Linear(bottleneck_size//4, self.num_points * 3)
        self.th = nn.Tanh()

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.th(self.fc4(x))
        x = x.view(batchsize, 3, self.num_points).transpose(1,2).contiguous()
        return x


class PointDecoderNormal(nn.Module):
    def __init__(self, num_points = 2048, bottleneck_size = 1024):
        super(PointDecoderNormal, self).__init__()
        self.num_points = num_points
        self.bottleneck_size = bottleneck_size
        self.bn1 = torch.nn.BatchNorm1d(bottleneck_size)
        self.bn2 = torch.nn.BatchNorm1d(bottleneck_size//2)
        self.bn3 = torch.nn.BatchNorm1d(bottleneck_size//4)
        self.fc1 = nn.Linear(self.bottleneck_size, bottleneck_size)
        self.fc2 = nn.Linear(self.bottleneck_size, bottleneck_size//2)
        self.fc3 = nn.Linear(bottleneck_size//2, bottleneck_size//4)
        self.fc4 = nn.Linear(bottleneck_size//4, self.num_points * 6)
        self.th = nn.Tanh()

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.th(self.fc4(x))
        x = x.view(batchsize, 6, self.num_points).transpose(1,2).contiguous()
        return x


class AE_Baseline(nn.Module):
    def __init__(self, num_points = 2048, bottleneck_size = 1024):
        super(AE_Baseline, self).__init__()
        self.num_points = num_points
        self.bottleneck_size = bottleneck_size
        self.encoder = nn.Sequential(
        PointNetfeat(num_points, global_feat=True, trans = False),
        nn.Linear(1024, self.bottleneck_size),
        nn.BatchNorm1d(self.bottleneck_size),
        nn.ReLU()
        )
        self.decoder = PointDecoder(num_points = num_points, bottleneck_size = self.bottleneck_size)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class AE_Baseline_normal(nn.Module):
    def __init__(self, num_points = 2048, bottleneck_size = 1024):
        super(AE_Baseline_normal, self).__init__()
        self.num_points = num_points
        self.bottleneck_size = bottleneck_size
        self.encoder = nn.Sequential(
        PointNetfeatNormal(num_points, global_feat=True, trans = False),
        nn.Linear(1024, self.bottleneck_size),
        nn.BatchNorm1d(self.bottleneck_size),
        nn.ReLU()
        )
        self.decoder = PointDecoderNormal(num_points = num_points, bottleneck_size = self.bottleneck_size)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class SVR_Baseline(nn.Module):
    def __init__(self, num_points = 2048, bottleneck_size = 1024, pretrained_encoder=False):
        super(SVR_Baseline, self).__init__()
        self.num_points = num_points
        self.bottleneck_size = bottleneck_size
        self.pretrained_encoder = pretrained_encoder
        self.encoder = resnet.resnet18(pretrained=self.pretrained_encoder,  num_classes=bottleneck_size)
        self.decoder = PointDecoder(num_points = num_points, bottleneck_size = self.bottleneck_size)

    def forward(self, x):
        x = x[:,:3,:,:].contiguous()

        x = self.encoder(x)
        x = self.decoder(x)
        return x


if __name__ == '__main__':
    print('Testing TangConv...')
    d  =  TangConvShapeNet(class_choice=None, balanced=False, train=True, npoints=2500)

    masks, normals, _, _ = d.__getitem__(50)

    # print(masks[0][1].shape)
    # input('Check Stats')

    # network = TangentConv(masks)
    network = TangConv_AtlasNet(num_points=2500, bottleneck_size=1024, nb_primitives=1)
    network.cuda()
    network.apply(weights_init)
    pointsReconstructed  = network(normals, masks)

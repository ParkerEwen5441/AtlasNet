from __future__ import print_function
import os
import sys
import torch
import os.path
import numpy as np
from utils import *
# from .utils.cloud import ScanData
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms

utils_path = '/home/parker/code/AtlasNet/utils/'
open3d_path = '/home/parker/packages/Open3D/build/lib/'
sys.path.append(utils_path)
sys.path.append(open3d_path)
from cloud import ScanData
from py3d import *

class TangConvShapeNet(data.Dataset):
    def __init__(self, root="/home/parker/datasets/TangConvNewTest", class_choice="couch",
                 train = True, npoints=2500, normal=False, balanced=False,
                 gen_view=False, SVR=False, idx=0, num_scales=3, max_points=2500,
                 input_channels=3):
        self.balanced = balanced
        self.normal = normal
        self.train = train
        self.root = root
        self.npoints = npoints
        self.datapath = []
        self.catfile = os.path.join('../data/synsetoffset2category.txt')
        self.cat = {}
        self.meta = {}
        self.SVR = SVR
        self.gen_view = gen_view
        self.idx=idx
        self.num_scales = num_scales
        self.training_data = []
        self.max_points = max_points
        self.input_channels = input_channels

        # From catfile, get chosen categories we want to use
        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]

        if not class_choice is  None:
            self.cat = {k:v for k,v in self.cat.items() if k in class_choice}
        print(self.cat)

        for item in self.cat:

            # Get directories of objects of a specific class in ShapeNetRendering folder
            dir_cat  = os.path.join(self.root, self.cat[item])
            fns = sorted(os.listdir(dir_cat))

            print('category: ', self.cat[item], 'files: ' + str(len(fns)))

            # First 20% of data for testing, last 80% for training
            if train:
                fns = fns[:int(len(fns) * 0.8)]
            else:
                fns = fns[int(len(fns) * 0.8):]


            # self.meta[item][0] = TangConv precompute directory
            #                [1] = ShapeNet point cloud file (.ply)
            #                [2] = name of the category of the item
            #                [3] = item name
            #
            #                [x] = path to the normalized_model dir in ShapeNetCorev2

            # For each non-matched item, remove it form self.cat
            if len(fns) != 0:
                self.meta[item] = []
                for fn in fns:
                    self.meta[item].append((os.path.join(dir_cat, fn),
                                            os.path.join(dir_cat, fn, fn + '.points.ply'),
                                            item, fn))

        self.idx2cat = {}
        self.size = {}

        # Stores self.meta[item] info into self.datapath[i]
        i = 0
        for item in self.cat:
            self.idx2cat[i] = item
            self.size[i] = len(self.meta[item])
            i = i + 1
            for fn in self.meta[item]:
                self.datapath.append(fn)


    def __getitem__(self, index):
        fn = self.datapath[index]

        # Reads each scale_x.npz file and extracts s.points, s.conv_ind, s.pool_ind,
        #   s.depth, and s.normals. Each scale represents a different layer size in the
        #   TangConv encoder.
        s = ScanData()
        s.load(fn[0], self.num_scales)
        s.remap_depth()
        s.remap_normals()

        if np.asarray(s.clouds[0].normals).shape[0] < self.max_points:
            s.resize(self.max_points)
        else:
            s.load(self.datapath[-2][0], self.num_scales)
            s.remap_depth()
            s.remap_normals()

        scale = []
        scale.append([np.asarray(s.clouds[0].points), s.conv_ind[0], s.pool_ind[0], s.depth[0], s.pool_mask[0]])
        scale.append([np.asarray(s.clouds[1].points), s.conv_ind[1], s.pool_ind[1], s.depth[1], s.pool_mask[1]])
        scale.append([np.asarray(s.clouds[2].points), s.conv_ind[2], s.pool_ind[2], s.depth[2], s.pool_mask[2]])

        normals = torch.from_numpy(np.asarray(s.clouds[0].normals))

        # return[0] : masks and other params for all scales
        #       [1] : Point set from point cloud .plc file
        #       [2] : category name
        #       [3] : item name
        #
        #       [x] : path to the normalized_model drir in ShapeNetCorev2

        # return data, point_set.contiguous(), fn[1], fn[2], fn[3]
        return scale, normals.contiguous(), fn[2], fn[3]


    def __len__(self):
        return len(self.datapath)


if __name__  == '__main__':

    d  =  TangConvShapeNet(class_choice =  None, balanced= False, train=True, npoints=2500)
    masks, normals, _, _ = d.__getitem__(50)
    print(masks[0][0].shape)
    print(masks[0][1].shape)
    print(masks[0][2].shape)
    print(masks[0][3].shape)
    print(masks[0][4].shape)
    print(normals.shape)

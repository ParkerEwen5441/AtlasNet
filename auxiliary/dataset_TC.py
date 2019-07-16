from __future__ import print_function
import os
import sys
import torch
import os.path
import numpy as np
from utils import *
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms

open3d_path = '/home/parker/packages/Open3D/build/lib/'
sys.path.append(open3d_path)
from py3d import *

class TangConvShapeNet(data.Dataset):
    def __init__(self, root="/home/parker/datasets/TangConv", class_choice="couch",
                 train = True, npoints=2500, normal=False, balanced=False,
                 gen_view=False, SVR=False, idx=0):
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
        self.num_scales = 3
        self.training_data = []

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
                                            os.path.join(dir_cat, fn + '.points.ply'),
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

        print('SAMPLE: {}'.format(self.datapath[0]))
        print(len(self.datapath))
        input("CHECK DATAPATH SIZE")

    def __getitem__(self, index):
        fn = self.datapath[index]

        # Opens file item .ply file and checks if it has information
        with open(fn[1]) as fp:
            for i, line in enumerate(fp):
                if i == 2:
                    try:
                        lenght = int(line.split()[2])
                    except ValueError:
                        print(fn)
                        print(line)
                    break

        # With .ply file, checks if it can load information as a float32
        # Iterates through the top loop 15 times to prevent a weird error. Lower loop gets
        #   n lines (n specified as self.npoints) from .ply file and stores as point_set
        #   given item
        for i in range(15): #this for loop is because of some weird error that happens sometime during loading I didn't track it down and brute force the solution like this.
            try:
                mystring = my_get_n_random_lines(fn[1], n = self.npoints)
                point_set = np.loadtxt(mystring).astype(np.float32)
                break
            except ValueError as excep:
                print(fn)
                print(excep)

        # Reads each scale_x.npz file and extracts s.points, s.conv_ind, s.pool_ind,
        #   s.depth, and s.normals. Each scale represents a different layer size in the
        #   TangConv encoder.
        s = ScanData()
        s.load(fn[0], self.num_scales)
        s.remap_depth()
        s.remap_normals()
        scale = [s.clouds[0], s.conv_ind[0], s.pool_ind[0], s.depth[0]]
        print('Cloud[0]: {}'.format(s.clouds[0]))
        print('Conv_ind[0]: {}'.format(s.conv_ind[0]))
        print('Pool_ind[0]: {}'.format(s.pool_ind[0]))
        print('Depth[0]: {}'.format(s.depth[0]))
        print('Depth[1]: {}'.format(s.depth[1]))
        print('Depth[2]: {}'.format(s.depth[2]))
        print('Scale: {}'.format(scale))
        input('CHECK FIRST INDEX')

        l = np.load(fn[0])
        cloud = PointCloud()
        cloud.points = Vector3dVector(l['points'])

        # If not normals, only keep first 3 points from .ply file. If normals, multiply
        #   normals from .ply by 0.1. Convert to pytorch tensor

        point_set = torch.from_numpy(point_set[:, 0:3])

        # return[0] : scale_0 masks and other params
        #       [1] : scale_0 masks and other params
        #       [2] : scale_0 masks and other params
        #       [3] : Point set from point cloud .plc file
        #       [3] : category name
        #       [4] : item name
        #
        #       [x] : path to the normalized_model drir in ShapeNetCorev2

        # return data, point_set.contiguous(), fn[1], fn[2], fn[3]
        return scale[0], scale[1], scale[2], point_set.contiguous(), fn[2], fn[3]


    def __len__(self):
        return len(self.datapath)



if __name__  == '__main__':

    print('Testing Shapenet dataset')
    d  =  TangConvShapeNet(class_choice =  None, balanced= False, train=True, npoints=2500)
    a = len(d)
    d  =  TangConvShapeNet(class_choice =  None, balanced= False, train=False, npoints=2500)
    a = a + len(d)
    print(a)

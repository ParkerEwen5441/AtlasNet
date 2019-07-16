from __future__ import print_function
import torch.utils.data as data
import os.path
import torch
import torchvision.transforms as transforms
import numpy as np
import os
from PIL import Image
from utils import *

open3d_path = '/home/parker/packages/Open3D/build/lib/'
sys.path.append(open3d_path)
from py3d import *

class TangConvShapeNet(data.Dataset):
    def __init__(self, root="/home/parker/datasets/ShapeNetTangConv", class_choice="car",
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

        # From catfile, get chosen categories we want to use
        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]

        if not class_choice is  None:
            self.cat = {k:v for k,v in self.cat.items() if k in class_choice}
        print(self.cat)

        empty = []
        for item in self.cat:

            # Get directories of objects of a specific class in ShapeNetRendering folder
            dir_img  = os.path.join(self.root, self.cat[item])
            fns_img = sorted(os.listdir(dir_img))

            # Matches items within the ShapeNetRendering and customShapeNet directories,
            #   only keeps item if it appears in both, prints out number of items kept
            fns = [val for val in fns_img]
            print('category: ', self.cat[item], 'files: ' + str(len(fns)))

            # First 20% of data for testing, last 80% for training
            if train:
                fns = fns[:int(len(fns) * 0.8)]
            else:
                fns = fns[int(len(fns) * 0.8):]


            # self.meta[item][0] = Tangent Image (.npz) from TangConv method
            #                [1] = ShapeNet point cloud file (.pcd)
            #                [2] = name of the category of the item
            #                [3] = file name
            #
            #                [x] = path to the normalized_model dir in ShapeNetCorev2

            # For each non-matched item, remove it form self.cat
            if len(fns) != 0:
                self.meta[item] = []
                for fn in fns:
                    self.meta[item].append((os.path.join(dir_img, fn, 'tangent_image.npz'),
                                            os.path.join(dir_img, fn + '.point_cloud.pcd'),
                                            item, fn))
            else:
                empty.append(item)

        for item in empty:
            del self.cat[item]

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

        # Normalization, data augmentation, and other transform functions for the .pngs
        # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                              std=[0.229, 0.224, 0.225])

        # self.transforms = transforms.Compose([
        #                      transforms.Resize(size =  224, interpolation = 2),
        #                      transforms.ToTensor(),
        #                      # normalize,
        #                 ])

        # # RandomResizedCrop or RandomCrop
        # self.dataAugmentation = transforms.Compose([
        #                                  transforms.RandomCrop(127),
        #                                  transforms.RandomHorizontalFlip(),
        #                     ])
        # self.validating = transforms.Compose([
        #                 transforms.CenterCrop(127),
        #                 ])

        # self.perCatValueMeter = {}

        # for item in self.cat:
        #     self.perCatValueMeter[item] = AverageValueMeter()

        # self.perCatValueMeter_metro = {}
        # for item in self.cat:
        #     self.perCatValueMeter_metro[item] = AverageValueMeter()

        # self.transformsb = transforms.Compose([
        #                      transforms.Resize(size =  224, interpolation = 2),
        #                 ])

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

        l = np.load(fn[0])
        cloud = PointCloud()
        cloud.points = Vector3dVector(l['points'])

        # If not normals, only keep first 3 points from .ply file. If normals, multiply
        #   normals from .ply by 0.1. Convert to pytorch tensor
        if not self.normal:
            point_set = point_set[:,0:3]
        else:
            point_set[:,3:6] = 0.1 * point_set[:,3:6]
        point_set = torch.from_numpy(point_set)

        # load image
        # if self.SVR:
        #     if self.train:
        #         N_tot = len(os.listdir(fn[0])) - 3
        #         if N_tot==1:
        #             print("only one view in ", fn)
        #         if self.gen_view:
        #             N=0
        #         else:
        #             N = np.random.randint(1,N_tot)
        #         if N < 10:
        #             im = Image.open(os.path.join(fn[0], "0" + str(N) + ".png"))
        #         else:
        #             im = Image.open(os.path.join(fn[0],  str(N) + ".png"))

        #         im = self.dataAugmentation(im) #random crop
        #     else:
        #         if self.idx < 10:
        #             im = Image.open(os.path.join(fn[0], "0" + str(self.idx) + ".png"))
        #         else:
        #             im = Image.open(os.path.join(fn[0],  str(self.idx) + ".png"))
        #         im = self.validating(im) #center crop
        #     data = self.transforms(im) #scale
        #     data = data[:3,:,:]
        # else:
        #     data = 0

        # return[0] : image if SVR, 0 otherwise
        #       [1] : Tangent Image file (.npz)
        #       [2] : Point Cloud file (.pcd)
        #       [3] : name of item category
        #       [4] : item name
        #
        #       [x] : path to the normalized_model drir in ShapeNetCorev2
        return data, point_set.contiguous(), fn[1], fn[2], fn[3]


    def __len__(self):
        return len(self.datapath)



if __name__  == '__main__':

    print('Testing Shapenet dataset')
    d  =  TangConvShapeNet(class_choice =  None, balanced= False, train=True, npoints=2500)
    a = len(d)
    d  =  TangConvShapeNet(class_choice =  None, balanced= False, train=False, npoints=2500)
    a = a + len(d)
    print(a)

import os
import subprocess
import numpy as np
import multiprocessing
open3d_path = '/home/parker/packages/Open3D/build/lib/'
sys.path.append(open3d_path)
from py3d import *


class ScanData():

    def __init__(self):
        self.clouds = []
        self.conv_ind = []
        self.pool_ind = []
        self.depth = []

    def load(self, file_path, num_scales):
        for i in range(0, num_scales):
            fname = os.path.join(file_path, "scale_" + str(i) + ".npz")
            l = np.load(fname)

            cloud = PointCloud()
            cloud.points = Vector3dVector(l['points'])

            estimate_normals(cloud, search_param=KDTreeSearchParamHybrid(radius=0.5, max_nn=100))

            self.clouds.append(cloud)
            self.conv_ind.append(l['nn_conv_ind'])
            self.depth.append(l['depth'])
            self.pool_ind.append(l['pool_ind'])

    def remap_depth(self, vmin=-1.0, vmax=1.0):
        num_scales = len(self.depth)
        for i in range(0, num_scales):
            self.depth[i] = np.clip(self.depth[i], vmin, vmax)
            self.depth[i] -= vmin
            self.depth[i] *= 1.0 / (vmax - vmin)

    def remap_normals(self, vmin=-1.0, vmax=1.0):
        num_scales = len(self.clouds)
        for i in range(0, num_scales):
            normals = np.asarray(self.clouds[i].normals)
            normals = np.clip(normals, vmin, vmax)
            normals -= vmin
            normals *= 1.0 / (vmax - vmin)
            self.clouds[i].normals = Vector3dVector(normals)

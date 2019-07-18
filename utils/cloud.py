import os
import sys
import subprocess
import numpy as np
import multiprocessing
open3d_path = '/home/parker/packages/Open3D/build/lib/'
sys.path.append(open3d_path)
from py3d import *

def get_pooling_mask(pooling):
    mask = np.asarray(pooling > 0, dtype='float32')
    sp = np.shape(mask)
    for i in range(0, sp[0]):
        mult = np.count_nonzero(mask[i, :])
        if mult == 0:
            mask[i, :] *= 0
        else:
            mask[i, :] *= 1/mult
    return mask


class ScanData():

    def __init__(self):
        self.clouds = []
        self.conv_ind = []
        self.pool_ind = []
        self.pool_mask = []
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
            self.pool_mask.append(get_pooling_mask(l['pool_ind']))

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

    def resize(self, max_points):
        '''
        Resizes all arrays to a set size depending on scale. Caused by discrepencies in
            sizes of point clouds.
            :param : maximum number of points allowed from point cloud sampling
                      (on the upper end of what is expected)
        '''
        num_scales = len(self.clouds)
        cloud = PointCloud()

        for i in range(0, num_scales):
            self.conv_ind[i] = self.conv_ind[i].transpose()
            self.depth[i] = self.depth[i].transpose()

            size = pow(2, i)
            num_points = max_points // size

            empty_normals = np.zeros([num_points, np.asarray(self.clouds[i].normals).shape[1]])
            empty_conv_ind = np.zeros([num_points, self.conv_ind[i].shape[1]])
            empty_depth = np.zeros([num_points, self.depth[i].shape[1]])
            empty_pool_ind = np.zeros([num_points, self.pool_ind[i].shape[1]])
            empty_pool_mask = np.zeros([num_points, self.pool_mask[i].shape[1]])

            empty_normals[:np.asarray(self.clouds[i].normals).shape[0],
                          :np.asarray(self.clouds[i].normals).shape[1]] = np.asarray(self.clouds[i].normals)
            empty_conv_ind[:self.conv_ind[i].shape[0], :self.conv_ind[i].shape[1]] = self.conv_ind[i]
            empty_depth[:self.depth[i].shape[0], :self.depth[i].shape[1]] = self.depth[i]
            empty_pool_ind[:self.pool_ind[i].shape[0], :self.pool_ind[i].shape[1]] = self.pool_ind[i]
            empty_pool_mask[:self.pool_mask[i].shape[0], :self.pool_mask[i].shape[1]] = self.pool_mask[i]

            cloud.normals = Vector3dVector(empty_normals)

            self.clouds[i].normals = cloud.normals
            self.conv_ind[i] = empty_conv_ind
            self.pool_ind[i] = empty_pool_ind
            self.pool_mask[i] = empty_pool_mask
            self.depth[i] = empty_depth

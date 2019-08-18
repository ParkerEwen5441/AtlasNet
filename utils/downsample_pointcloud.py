import os
import sys
import glob
import argparse
import numpy as np
import open3d as o3d


parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, default='', help='directory of dataset')
options = parser.parse_args()

DATASET_DIR = options.dir


def find_ply():
    '''
    Walk through TangConv dataset directories to find all .ply files to downsample
    '''

    for _, classdirs, _ in os.walk(DATASET_DIR):
        for i in range(len(classdirs)):
            for _, _, files in os.walk(os.path.join(DATASET_DIR, classdirs[i], 'ply')):
                for j in range(len(files)):
                        if not "*" in files[j] and not "txt" in files[j]:
                            item_class = classdirs[i]
                            filename = os.path.join(DATASET_DIR, classdirs[i], 'ply', files[j])
                            downsample(item_class, filename)



def downsample(item_class, file):
    '''
    Use Open3D voxel downsampling to reduce number of points to below 2500
    :param item_class: item class the ply belongs to
           file      : directory including file name of original ply file
    '''

    ### IMPLEMENT INCREMENTAL VOXEL SIZE INCREASE FOR OPTIMAL NUMBER OF POINTS
    pcd = o3d.io.read_point_cloud(file)
    voxel_size = 0.05
    # last_voxel_size = 0.05
    # try_smaller = True

    # while try_smaller:
    #     downpcd = o3d.geometry.voxel_down_sample(pcd, voxel_size=voxel_size)
    #     if np.asarray(downpcd.points).shape[0] <= 2500:
    #         last_voxel_size = voxel_size
    #         voxel_size = voxel_size - 0.01
    #     elif np.asarray(downpcd.points).shape[0] > 2500 and voxel_size >= 0.05:
    #         last_voxel_size = voxel_size
    #         voxel_size = voxel_size + 0.01
    #     else:
    #         downpcd = o3d.geometry.voxel_down_sample(pcd, voxel_size=last_voxel_size)
    #         try_smaller = False

    downpcd = o3d.geometry.voxel_down_sample(pcd, voxel_size=voxel_size)

    if np.asarray(downpcd.points).shape[0] > 2500:
        go_smaller = True
        while go_smaller:
            voxel_size = voxel_size + 0.01
            downpcd = o3d.geometry.voxel_down_sample(pcd, voxel_size=voxel_size)
            if np.asarray(downpcd.points).shape[0] <= 2500:
                go_smaller = False

    dataset = '/home/parker/datasets/TangConvDownSample'
    item = file.split('/')[-1]
    newdir = os.path.join(dataset, item_class, 'ply')
    newfile = os.path.join(newdir, item)

    if not os.path.exists(newdir):
        os.makedirs(newdir)

    o3d.io.write_point_cloud(newfile, downpcd)


if __name__ == "__main__":
    find_ply()

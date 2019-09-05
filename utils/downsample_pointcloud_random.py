import os
import sys
import glob
import random
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

    for i in range(15): #this for loop is because of some weird error that happens sometime during loading I didn't track it down and brute force the solution like this.
        try:
            mystring = my_get_n_random_lines(file, n=2500)
            point_set = np.loadtxt(mystring).astype(np.float32)
            break
        except ValueError as excep:
            print(file)
            print(excep)

    points = point_set[:, 0:3]
    normals = point_set[:, 3:6]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.normals = o3d.utility.Vector3dVector(normals)

    dataset = '/home/parker/datasets/TangConvRandDS'
    item = file.split('/')[-1]
    newdir = os.path.join(dataset, item_class, 'ply')
    newfile = os.path.join(newdir, item)

    if not os.path.exists(newdir):
        os.makedirs(newdir)

    # pcd_load = o3d.io.read_point_cloud(file)
    # o3d.visualization.draw_geometries([pcd])
    # o3d.visualization.draw_geometries([pcd_load])

    o3d.io.write_point_cloud(newfile, pcd)


def my_get_n_random_lines(path, n=2500):
    lenght_line = 60
    MY_CHUNK_SIZE = lenght_line * (n+2)
    lenght = os.stat(path).st_size

    with open(path, 'r') as file:
        file.seek(random.randint(400, lenght - MY_CHUNK_SIZE))
        chunk = file.read(MY_CHUNK_SIZE)
        lines = chunk.split(os.linesep)
        return lines[1:n+1]


if __name__ == "__main__":
    find_ply()

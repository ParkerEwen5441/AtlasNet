import os
import random
import string
import shutil
import numpy as np

ROOT_DIR = '/home/parker/datasets/ShapeNetTangConv'
PC_DIR = '/home/parker/datasets/ShapeNetPointCloud'
CAT_FILE = '/home/parker/code/AtlasNet/data/synsetoffset2category.txt'

def categories():
    '''
    Gets dictionary of all categories used in ShapeNet
    '''
    cat = {}
    with open(CAT_FILE, 'r') as f:
        for line in f:
            ls = line.strip().split()
            cat[ls[0]] = ls[1]

    return cat


def get_files():
    '''
    Finds all TangConv files written to ROOT directory
    '''
    list_of_files = []
    for _, item_dirs, _ in os.walk(ROOT_DIR):
        for item in item_dirs:
            ITEM_DIR = os.path.join(ROOT_DIR, item)
            for _, scale_dirs, _ in os.walk(ITEM_DIR):
                for scale in scale_dirs:
                    SCALE_DIR = os.path.join(ITEM_DIR, scale)
                    for _, _, files in os.walk(SCALE_DIR):
                        for file in files:
                            list_of_files.append(os.path.join(SCALE_DIR, file))

    return list_of_files


def prelim_reorganize(cat):
    '''
    Makes category directories in ROOT if they don't exist
    :param : cat, dictionary of category names and corresponding numbers
    '''
    for key, value in cat.items():
        if not os.path.exists(os.path.join(ROOT_DIR, value)):
            os.mkdir(os.path.join(ROOT_DIR, value))


def reorganize(cat, list_of_files):
    keys = list(cat.keys())

    for file in list_of_files:
        for key in keys:
            if key in file.split('/')[5]:
                rand_name = ''.join(random.choice(string.ascii_uppercase + string.digits)
                                    for _ in range(10))
                new_folder = os.path.join(ROOT_DIR, cat[key], rand_name)
                os.mkdir(new_folder)
                pc_file = os.path.join(PC_DIR, key, file.split('/')[5])

                if os.path.exists(pc_file):
                    # print(os.path.join(new_folder, new_file))
                    # print(os.path.join(PC_DIR, pc_file))
                    shutil.move(file, os.path.join(new_folder, 'tangent_image.npz'))
                    shutil.move(os.path.join(PC_DIR, pc_file), os.path.join(new_folder, 'point_cloud.pcd'))


def main():
    cat = categories()
    list_of_files = get_files()
    prelim_reorganize(cat)
    reorganize(cat, list_of_files)


if __name__ == '__main__':
    main()

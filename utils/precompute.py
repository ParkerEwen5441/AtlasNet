import os
import sys
import math
import json
import shutil
import inspect
import numpy as np
import open3d as o3d

open3d_path = '/home/parker/packages/Open3D/build/lib/'
tc_path = '/home/parker/code/AtlasNet/'
sys.path.append(open3d_path)

from py3d import *

def get_tc_path():
	return tc_path

class param:
	def __init__(self):
		with open('../config/config.json') as f:
			config = json.load(f)

		self.min_cube_size = config['pre_min_cube_size']
		self.filter_size = config['pre_filter_size']
		self.num_scales = config['pre_num_scales']
		self.num_rotations = config['pre_num_rotations']
		self.num_neighbors = config['pre_num_neighbors']
		self.dataset_dir = config['pre_dataset_dir']
		self.output_dir = config['pre_output_dir']
		self.interp_method = config['pre_interp_method']
		self.noise_level = config['pre_noise_level']

def run_precompute():
	p = param()
	make_dir(p.output_dir)

	for cat in list_dir(p.dataset_dir):
		for scan_name in list_files(os.path.join(p.dataset_dir, cat, 'ply')):
			if '.txt' in scan_name:
				continue

			cloud_file = os.path.join(p.dataset_dir, cat, 'ply', scan_name)

			if not os.path.exists(cloud_file):
				continue

			print("processing scan: %s" % (scan_name))

			pcd_colors = read_point_cloud(cloud_file)
			pcd_colors.points = Vector3dVector(np.asarray(pcd_colors.points))

			min_bound = pcd_colors.get_min_bound() - p.min_cube_size * 0.5
			max_bound = pcd_colors.get_max_bound() + p.min_cube_size * 0.5

			for i in range(0, p.num_scales):
				multiplier = pow(2, i)
				pcd_colors_down = voxel_down_sample_for_surface_conv(pcd_colors, multiplier*p.min_cube_size,
					min_bound, max_bound, False)

				method = depth_densify_nearest_neighbor

				parametrization = planar_parametrization(pcd_colors_down.point_cloud,
							KDTreeSearchParamHybrid(radius = 2*multiplier*p.min_cube_size, max_nn=100),
							PlanarParameterizationOption(
							sigma = 1, number_of_neighbors=p.num_neighbors, half_patch_size=p.filter_size//2,
							depth_densify_method = method))

				item_name = scan_name.split('.')[0]
				if not os.path.exists(os.path.join(p.output_dir, cat)):
					os.mkdir(os.path.join(p.output_dir, cat))
				if not os.path.exists(os.path.join(p.output_dir, cat, item_name)):
					os.mkdir(os.path.join(p.output_dir, cat, item_name))

				np.savez_compressed(os.path.join(p.output_dir, cat, item_name, 'scale_' + str(i) + '.npz'),
						points=np.asarray(pcd_colors_down.point_cloud.points),
						nn_conv_ind=parametrization.index[0],
						pool_ind=pcd_colors_down.cubic_id,
						depth=parametrization.depth.data)

				sys.stdout.flush()

			shutil.copy(cloud_file, os.path.join(p.output_dir, cat, item_name))
			sys.stdout.write("\n")


def read_txt_labels(file_path):
	with open(file_path) as f:
		labels = f.readlines()
		lb = [c.rstrip() for c in labels]
	return np.asarray(lb, dtype='int32')


def make_dir(directory):
	if not os.path.exists(directory):
		os.makedirs(directory)


def list_dir(directory):
	paths = []
	for root, dirs, files in os.walk(directory):
		for dirr in dirs:
			paths.append(dirr)
	return paths

def list_files(directory):
	paths = []
	for root, dirs, files in os.walk(directory):
		for file in files:
			paths.append(file)
	return paths


if __name__ == "__main__":
	run_precompute()

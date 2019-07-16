import os
import sys
import math
import json
import inspect
import numpy as np

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

	for scan_name in list_files(p.dataset_dir):
		if '.txt' in scan_name:
			continue
		for rot in range(0, p.num_rotations):
			# cloud_file = os.path.join(p.dataset_dir, scan_name, "scan.pcd")
			cloud_file = os.path.join(p.dataset_dir, scan_name)

			# label_file = os.path.join(p.dataset_dir, scan_name, "scan.labels")

			# if not os.path.exists(cloud_file) or not os.path.exists(label_file):
			# 	continue
			if not os.path.exists(cloud_file):
				continue

			if p.noise_level != 0.0:
				print("processing scan: %s, rot: %d, noise: %f" %
						(scan_name, rot, p.noise_level))
			else:
				print("processing scan: %s, rot: %d" % (scan_name, rot))

			pcd_colors = read_point_cloud(cloud_file)
			pcd_labels = read_point_cloud(cloud_file)
			# txt_labels = read_txt_labels(label_file)

			theta = rot * 2 * math.pi / p.num_rotations

			rot_matrix = np.asarray(
						   [[np.cos(theta), -np.sin(theta), 0],
							[np.sin(theta), np.cos(theta), 0],
							[0, 0, 1]])
			pcd_colors.points = Vector3dVector(np.matmul(np.asarray(pcd_colors.points),
											 			 rot_matrix))
			pcd_labels.points = Vector3dVector(np.matmul(np.asarray(pcd_labels.points),
											 			 rot_matrix))

			# additive gauissan noise
			if p.noise_level != 0.0:
				npts = len(pcd_colors.points)
				additive_noise_to_points = np.random.normal(0.0, p.noise_level, (npts,3))
				for i in range(npts):
					pcd_colors.points[i] += additive_noise_to_points[i]
					pcd_labels.points[i] += additive_noise_to_points[i]

			# lb = np.repeat(np.expand_dims(txt_labels, axis=1), 3, axis=1)
			# pcd_labels.colors = Vector3dVector(lb)

			min_bound = pcd_colors.get_min_bound() - p.min_cube_size * 0.5
			max_bound = pcd_colors.get_max_bound() + p.min_cube_size * 0.5

			make_dir(os.path.join(p.output_dir, scan_name))
			make_dir(os.path.join(p.output_dir, scan_name, str(rot)))

			for i in range(0, p.num_scales):
				multiplier = pow(2, i)
				pcd_colors_down = voxel_down_sample_for_surface_conv(pcd_colors, multiplier*p.min_cube_size,
					min_bound, max_bound, False)
				# pcd_labels_down = voxel_down_sample_for_surface_conv(pcd_labels, multiplier*p.min_cube_size,
				# 	min_bound, max_bound, True)

				if p.interp_method == "depth_densify_nearest_neighbor":
					method = depth_densify_nearest_neighbor
				elif p.interp_method == "depth_densify_gaussian_kernel":
					method = depth_densify_gaussian_kernel

				parametrization = planar_parametrization(pcd_colors_down.point_cloud,
							KDTreeSearchParamHybrid(radius = 2*multiplier*p.min_cube_size, max_nn=100),
							PlanarParameterizationOption(
							sigma = 1, number_of_neighbors=p.num_neighbors, half_patch_size=p.filter_size//2,
							depth_densify_method = method))

				num_points = np.shape(np.asarray(pcd_colors_down.point_cloud.points))[0]

				np.savez_compressed(os.path.join(p.output_dir, scan_name, str(rot), 'scale_' + str(i) + '.npz'),
						points=np.asarray(pcd_colors_down.point_cloud.points),
						# colors=np.asarray(pcd_colors_down.point_cloud.colors),
						# labels_gt=np.reshape(np.asarray(pcd_labels_down.point_cloud.colors)[:, 0], (num_points)),
						nn_conv_ind=parametrization.index[0],
						pool_ind=pcd_colors_down.cubic_id,
						depth=parametrization.depth.data)

				# TODO Add support for Gaussian kernel

				pcd_colors = pcd_colors_down.point_cloud
				# pcd_labels = pcd_labels_down.point_cloud


				data = np.load(os.path.join(p.output_dir, scan_name, str(rot), 'scale_' + str(i) + '.npz'))
				lst = data.files
				# for item in lst:
				#     print(item)
				#     print(data[item])
				    # input("WAIT")
				print(data['points'].shape)
				print(data['nn_conv_ind'].shape)
				print(data['pool_ind'].shape)
				print(data['depth'].shape)
				print(data['points'])
				print(data['nn_conv_ind'])
				print(data['pool_ind'])
				input("CHECK HERE FOR TANG IMAGE")


			sys.stdout.flush()
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
		for subdir in dirs:
			paths.append(os.path.join(root.replace(directory,''), subdir))
	return paths

def list_files(directory):
	paths = []
	for root, dirs, files in os.walk(directory):
		for file in files:
			paths.append(os.path.join(root.replace(directory,''), file))
	return paths


if __name__ == "__main__":
	run_precompute()

/*
 * Software License Agreement (BSD License)
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2010-2011, Willow Garage, Inc.
 *
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the copyright holder(s) nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 */

#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/transforms.h>
#include <vtkPLYReader.h>
#include <vtkOBJReader.h>
#include <vtkPolyDataMapper.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/console/print.h>
#include <pcl/console/parse.h>
#include <fstream>
#include <sstream>
#include <string>
#include <unistd.h>

// using namespace pcl;
// using namespace pcl::io;
// using namespace pcl::console;

int default_tesselated_sphere_level = 2;
int default_resolution = 100;
float default_leaf_size = 0.01f;
std::string SHAPENET_DIR = "/home/parker/datasets/ShapeNetCore.v2";
std::string SUB_PATH = "models/model_normalized.obj";
std::string PATHS_FILE = "config/obj_paths.txt";
std::string OUT_DIR = "/home/parker/datasets/ShapeNetPointCloud";

void printHelp (int, char **argv)
{
  pcl::console::print_error ("Syntax is: %s input.{ply,obj} output.pcd <options>\n", argv[0]);
  pcl::console::print_info ("  where options are:\n");
  pcl::console::print_info ("                     -level X      = tesselated sphere level (default: ");
  pcl::console::print_value ("%d", default_tesselated_sphere_level);
  pcl::console::print_info (")\n");
  pcl::console::print_info ("                     -resolution X = the sphere resolution in angle increments (default: ");
  pcl::console::print_value ("%d", default_resolution);
  pcl::console::print_info (" deg)\n");
  pcl::console::print_info (
              "                     -leaf_size X  = the XYZ leaf size for the VoxelGrid -- for data reduction (default: ");
  pcl::console::print_value ("%f", default_leaf_size);
  pcl::console::print_info (" m)\n");
}

void obj2pcd (pcl::visualization::PCLVisualizer::Ptr& vis, const std::string& in, const std::string& out,
              int tesselated_sphere_level, int resolution, float leaf_size,
              bool INTER_VIS = false, bool VIS = false)
{
  vtkSmartPointer<vtkPolyData> polydata1;
  vtkSmartPointer<vtkOBJReader> readerQuery = vtkSmartPointer<vtkOBJReader>::New ();
  readerQuery->SetFileName (in.c_str());
  // polydata1 = readerQuery->GetOutput ();
  // polydata1->Update ();
  readerQuery->Update ();
  polydata1 = readerQuery->GetOutput ();

  // pcl::visualization::PCLVisualizer::Ptr vis(new pcl::visualization::PCLVisualizer ("viewer"));
  vis->addModelFromPolyData (polydata1, "mesh1", 0);
  vis->setRepresentationToSurfaceForAllActors ();

  pcl::PointCloud<pcl::PointXYZ>::CloudVectorType views_xyz;
  std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > poses;
  std::vector<float> enthropies;
  vis->renderViewTesselatedSphere (resolution, resolution, views_xyz, poses, enthropies, tesselated_sphere_level);

  //take views and fuse them together
  std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> aligned_clouds;

  for (size_t i = 0; i < views_xyz.size (); i++)
  {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ> ());
    Eigen::Matrix4f pose_inverse;
    pose_inverse = poses[i].inverse ();
    pcl::transformPointCloud (views_xyz[i], *cloud, pose_inverse);
    aligned_clouds.push_back (cloud);
  }

  // if (INTER_VIS)
  // {
  //   visualization::PCLVisualizer vis2 ("visualize");

  //   for (size_t i = 0; i < aligned_clouds.size (); i++)
  //   {
  //     std::stringstream name;
  //     name << "cloud_" << i;
  //     vis2.addPointCloud (aligned_clouds[i], name.str ());
  //     vis2.spin ();
  //   }
  // }

  // Fuse clouds
  pcl::PointCloud<pcl::PointXYZ>::Ptr big_boy (new pcl::PointCloud<pcl::PointXYZ> ());
  for (size_t i = 0; i < aligned_clouds.size (); i++)
    *big_boy += *aligned_clouds[i];

  // if (VIS)
  // {
  //   visualization::PCLVisualizer vis2 ("visualize");
  //   vis2.addPointCloud (big_boy);
  //   vis2.spin ();
  // }

  // Voxelgrid
  pcl::VoxelGrid<pcl::PointXYZ> grid_;
  grid_.setInputCloud (big_boy);
  grid_.setLeafSize (leaf_size, leaf_size, leaf_size);
  grid_.filter (*big_boy);

  // if (VIS)
  // {
  //   visualization::PCLVisualizer vis3 ("visualize");
  //   vis3.addPointCloud (big_boy);
  //   vis3.spin ();
  // }

  pcl::io::savePCDFileASCII (out.c_str(), *big_boy);

  vis->spinOnce();
  // usleep(100000);
  // vis->close();
  // vis.reset();
}

/* ---[ */
int main (int argc, char **argv)
{
  pcl::console::print_info ("Convert a CAD model to a point cloud using ray tracing operations. For more information, use: %s -h\n",
              argv[0]);

  // if (argc < 3)
  // {
  //   printHelp (argc, argv);
  //   return (-1);
  // }

  // Parse command line arguments
  int tesselated_sphere_level = default_tesselated_sphere_level;
  pcl::console::parse_argument (argc, argv, "-level", tesselated_sphere_level);
  int resolution = default_resolution;
  pcl::console::parse_argument (argc, argv, "-resolution", resolution);
  float leaf_size = default_leaf_size;
  pcl::console::parse_argument (argc, argv, "-leaf_size", leaf_size);

  std::ifstream infile("config/obj_paths.txt");

  pcl::visualization::PCLVisualizer::Ptr vis(new pcl::visualization::PCLVisualizer ("viewer"));

  int i = 3712;

  std::string line;
  while (std::getline(infile, line))
  {
    std::istringstream iss(line);
    std::string class_name, obj_path;
    if (!(iss >> class_name >> obj_path)) { break; } // error

    std::stringstream ss;
    ss << OUT_DIR << "/" << class_name << "/" << class_name << std::setw(5) << std::setfill('0') << i << ".pcd";
    std::string out_file = ss.str();

    std::cout << "Converting " << class_name << " from " << obj_path << " to " << out_file << std::endl;

    obj2pcd(vis, obj_path, out_file, tesselated_sphere_level, resolution, leaf_size);

    vis->removeAllPointClouds();
    vis->removeAllShapes();

    i++;
  }
}

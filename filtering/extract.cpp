#include <pcl/ModelCoefficients.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/visualization/cloud_viewer.h>
#include <string>

 
int main(int argc, char** argv)
{

	string home_path = "/home/drla/ml3d/pySceneNetRGBD/data/val/0/";

	// 1000 scenes
	for(int index_trajectory=0;index_trajectory<1000;index_trajectory++){

		string dir_trajectory = home_path + to_string(index_trajectory);

		// 7475 is the max index for images, interval is 25
		for(int index_image=0;index_image<=7475;index_image += 25){
		// Read in the cloud data
			pcl::PCDReader reader;
			pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>), cloud_f_1(new pcl::PointCloud<pcl::PointXYZ>), cloud_f(new pcl::PointCloud<pcl::PointXYZ>);
			
			string pcd_before = dir_trajectory + to_string(index_image) + ".pcd"
			reader.read(pcd_before, *cloud);
			std::cout << "PointCloud before filtering has: " << cloud->points.size() << " data points." << std::endl; //*
		
		
		// Create the filtering object: downsample the dataset using a leaf size of 1cm
			pcl::VoxelGrid<pcl::PointXYZ> vg;
			pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
			vg.setInputCloud(cloud);
			vg.setLeafSize(0.01f, 0.01f, 0.01f);
			vg.filter(*cloud_filtered);
			std::cout << "PointCloud after filtering has: " << cloud_filtered->points.size() << " data points." << std::endl; //*
			pcl::copyPointCloud(*cloud_filtered, *cloud_f_1);
		
		
		// Create the segmentation object for the planar model and set all the parameters
			pcl::SACSegmentation<pcl::PointXYZ> seg;
			pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
			pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
			pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_plane(new pcl::PointCloud<pcl::PointXYZ>());
			pcl::PCDWriter writer;
			seg.setOptimizeCoefficients(true);
			seg.setModelType(pcl::SACMODEL_PLANE);
			seg.setMethodType(pcl::SAC_RANSAC);
			seg.setMaxIterations(100);
			seg.setDistanceThreshold(0.02);
		
		
			int i = 0, nr_points = (int)cloud_filtered->points.size();
			while (cloud_filtered->points.size() > 0.1 * nr_points)
			{
				// Segment the largest planar component from the remaining cloud
				seg.setInputCloud(cloud_filtered);
				seg.segment(*inliers, *coefficients);
				if (inliers->indices.size() == 0)
				{
					std::cout << "Could not estimate a planar model for the given dataset." << std::endl;
					break;
				}
		
		
				// Extract the planar inliers from the input cloud
				pcl::ExtractIndices<pcl::PointXYZ> extract;
				extract.setInputCloud(cloud_filtered);
				extract.setIndices(inliers);
				extract.setNegative(false);
		
		
				// Write the planar inliers to disk
				extract.filter(*cloud_plane);
				std::cout << "PointCloud representing the planar component: " << cloud_plane->points.size() << " data points." << std::endl;
		
		
				// Remove the planar inliers, extract the rest
				extract.setNegative(true);
				extract.filter(*cloud_f);
				cloud_filtered = cloud_f;
			}
		
		
			boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
		
			/*
			int v1(0);
			viewer->createViewPort(0.0, 0.0, 0.5, 1.0, v1);
			viewer->setBackgroundColor(0, 0, 0, v1);
			viewer->addText("Radius: 0.01", 10, 10, "v1 text", v1);
			viewer->addPointCloud<pcl::PointXYZ>(cloud_f_1,"sample cloud1", v1);
		
		
			int v2(0);
			viewer->createViewPort(0.5, 0.0, 1.0, 1.0, v2);
			viewer->setBackgroundColor(0.3, 0.3, 0.3, v2);
			viewer->addText("Radius: 0.1", 10, 10, "v2 text", v2);
			viewer->addPointCloud<pcl::PointXYZ>(cloud_filtered, "sample cloud2", v2);

			viewer->initCameraParameters();
		
			
			while (!viewer->wasStopped())
			{
				viewer->spinOnce(100);
				boost::this_thread::sleep(boost::posix_time::microseconds(100000));
			}
		
			*/

			// output .pcd file without walls, floors and ceilings
			int points_len = cloud_filtered->points.size();

			string pcd_after = dir_trajectory + to_string(index_image) + "_after_filtering.pcd"
			std::string pc_name = pcd_after;

			FILE *fp = fopen(pc_name.c_str(), "wb");
			fprintf(fp, "# .PCD v0.7 - Point Cloud Data file format\n");
			fprintf(fp, "VERSION 0.7\n");
			fprintf(fp, "FIELDS x y z\n");
			fprintf(fp, "SIZE 4 4 4\n");
			fprintf(fp, "TYPE F F F\n");
			fprintf(fp, "COUNT 1 1 1\n");
			fprintf(fp, "WIDTH %d\n", points_len);
			fprintf(fp, "HEIGHT 1\n");
			fprintf(fp, "VIEWPOINT 0 0 0 1 0 0 0\n");
			fprintf(fp, "POINTS %d\n", points_len);
			fprintf(fp, "DATA binary\n");

			for (int n = 0; n < points_len; n++) {
				float tmp2[3] = { cloud_filtered->points[n].x, cloud_filtered->points[n].y, cloud_filtered->points[n].z };
				fwrite(tmp2, sizeof(float), 3, fp);
			}
			fclose(fp);
		}
	}
	return (0);
}

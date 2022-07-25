#include <math.h>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <string>

using namespace std;

double degree2radians(double degree){
    return degree*M_PI / 180;
}


pcl::PointCloud<pcl::PointXYZRGB>::Ptr depth2cloud(cv::Mat rgb, cv::Mat depth)
{
    // double fx = 277.12812921, fy=289.70562748;
    // double cx = 160.0, cy = 120.0;

    double depthscale = 1000;

    double vfov=45, hfov=60, pixel_width=320, pixel_height=240;

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_ptr( new pcl::PointCloud<pcl::PointXYZRGB> () );
    cloud_ptr->width  = rgb.cols;
    cloud_ptr->height = rgb.rows;
    cloud_ptr->is_dense = false;

    for ( int y = 0; y < rgb.rows; ++ y ) {
        for ( int x = 0; x < rgb.cols; ++ x ) {
        pcl::PointXYZRGB pt;
        if ( depth.at<unsigned short>(y, x) != 0 )
        {   
            //transform to normalized plane
            double x_vect = tan(degree2radians(hfov/2.0))*((2.0*((x+0.5)/pixel_width))-1.0);
            double y_vect = tan(degree2radians(vfov/2.0))*((2.0*((y+0.5)/pixel_height))-1.0);
            double z_vect = 1;

            double norm = sqrt(pow(x_vect,2)+pow(y_vect,2)+1);

            double x_normalized = x_vect / norm;
            double y_normalized = y_vect / norm;
            double z_normalized = z_vect / norm;

            // pt.z = (255-depth.at<unsigned short>(y, x))/depthscale;
            // pt.x = (x-cx)*pt.z/fx;
            // pt.y = (y-cy)*pt.z/fy;
            pt.x = x_normalized * depth.at<unsigned short>(y,x) / depthscale;
            pt.y = y_normalized * depth.at<unsigned short>(y,x) / depthscale;
            pt.z = z_normalized * depth.at<unsigned short>(y,x) / depthscale;

            pt.r = rgb.at<cv::Vec3b>(y, x)[2];
            pt.g = rgb.at<cv::Vec3b>(y, x)[1];
            pt.b = rgb.at<cv::Vec3b>(y, x)[0];
            cloud_ptr->points.push_back( pt );
        }
        else
        {
            pt.z = 0;
            pt.x = 0;
            pt.y = 0;
            pt.r = 0;
            pt.g = 0;
            pt.b = 0;
            cloud_ptr->points.push_back( pt );
        }
    }
  }
  return cloud_ptr;

}

 
int main( int argc, char** argv ){

    string home_path = "../scenenet/val/0/";
    
    
    int index_trajectory;

    //change to index_trajectory<100
    for(index_trajectory=0;index_trajectory<150;index_trajectory++){

        string dir_trajectory = home_path + to_string(index_trajectory); //计算每一个轨迹目录
        
        string dir_depth = dir_trajectory + "/depth/";
        string dir_photo = dir_trajectory+ "/photo/";

        cout<<dir_depth<<endl;
        cout<<dir_photo<<endl;

        int index_image;
        //change to index_image<=7475
        for(index_image=0;index_image<=7475;index_image += 25){

            cv::Mat rgb, depth;

            string depth_image = dir_depth + to_string(index_image)+".png";
            string photo_image = dir_photo + to_string(index_image)+".jpg";

            cout<<depth_image<<endl;
            cout<<photo_image<<endl;

            rgb = cv::imread(photo_image);
            depth = cv::imread(depth_image,-1);

            if(!rgb.data||!depth.data)        // 判断图片调入是否成功
                return -1;        // 调入图片失败则退出

            pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
        
            cloud=depth2cloud(rgb,depth);

            pcl::io::savePCDFileBinary(dir_trajectory + "/" + to_string(index_image) + ".pcd",*cloud);

            std::cout<<"Successfully generate"<<to_string(index_image)<<std::endl;
        
        }
    }
    

    return 0;

}


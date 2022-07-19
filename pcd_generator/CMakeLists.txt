# cmake版本及工程名
cmake_minimum_required( VERSION 2.8 )
project( rgbd2pcd )
 
# 设置opencv库
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
 
# 设置pcl库 
find_package( PCL REQUIRED)
include_directories( ${PCL_INCLUDE_DIRS} )
 
# 编译与链接
add_executable( rgbd2pcd rgbd2pcd.cpp )
target_link_libraries( rgbd2pcd ${OpenCV_LIBS} ${PCL_LIBRARIES} )

import open3d as o3d
import numpy as np
import argparse


def pcd2pointcloud(pcd_file, pointcloud_file):
    pcd = o3d.io.read_point_cloud(pcd_file)
    print(pcd.has_colors)
    points = np.array(pcd.points)
    point_cloud = ''
    file = open(pointcloud_file, 'w+')
    for i, point in enumerate(points):
        point_cloud = point_cloud + str(point[0]) + " " + str(point[1]) + " " + str(point[2]) + " " \
            + str(pcd.colors[i][0]*255) + ' ' + str(pcd.colors[i][1]*255) + ' ' + str(pcd.colors[i][2]*255) +'\n'

    file.write(point_cloud)
    file.close()

def main():
    parser = argparse.ArgumentParser(description='PCD file transformation')
    parser.add_argument('--pcd', type=str, default='./pcd_file.pcd', metavar='N',
                        help='Name of the pcd file')
    parser.add_argument('--point_cloud', type=str, default='./point_cloud.txt', metavar='N',
                        help='Name of the point cloud file')
    args = parser.parse_args()
    pcd2pointcloud(args.pcd, args.point_cloud)


if __name__ == '__main__':
    main()
import open3d as o3d
import numpy as np
import argparse
import tqdm


def pcd2pointcloud(pcd_path, pointcloud_file):
    for i in range(1):
        print('---------------- loading pcd file '+str(i)+ '----------------')
        for p in range(21):
            pcd = o3d.io.read_point_cloud(pcd_path +'\\'+ str(i) + '\\'+ str(p*25)+'.pcd')
            points = np.array(pcd.points)
            point_cloud = ''
            file = open(pointcloud_file+str(i)+'_'+str(p*25)+'.txt', 'w+')
            for k, point in enumerate(points):
                point_cloud = point_cloud + str(point[0]) + " " + str(point[1]) + " " + str(point[2]) + " " \
                    + str(pcd.colors[k][0]*255) + ' ' + str(pcd.colors[k][1]*255) + ' ' + str(pcd.colors[k][2]*255) +'\n'

            file.write(point_cloud)
            file.close()
            print(pointcloud_file+str(i)+'_'+str(p)+'.txt is successfully created.')

def main():
    parser = argparse.ArgumentParser(description='PCD file transformation')
    parser.add_argument('--pcd', type=str, default='./pcd/', metavar='N',
                        help='Name of the pcd path')
    parser.add_argument('--point_cloud', type=str, default='./point_cloud.txt', metavar='N',
                        help='Name of the point cloud file')
    args = parser.parse_args()
    pcd2pointcloud(args.pcd, args.point_cloud)


if __name__ == '__main__':
    main()

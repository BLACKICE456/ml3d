import numpy as np
import os

abs_path = os.path.abspath(os.path.dirname(__file__))+'/scenenet/ground_truth/'

def normalize(v):
    return v/np.linalg.norm(v)

def world_to_camera_with_pose(view_pose):
    lookat_pose = view_pose['lookat']
    camera_pose = view_pose['camera']
    up = np.array([0,1,0])
    R = np.diag(np.ones(4))
    R[2,:3] = normalize(lookat_pose - camera_pose)
    R[0,:3] = normalize(np.cross(R[2,:3],up))
    R[1,:3] = -normalize(np.cross(R[0,:3],R[2,:3]))
    T = np.diag(np.ones(4))
    T[:3,3] = -camera_pose
    return R.dot(T)


def camera_to_world_with_pose(view_pose):
    return np.linalg.inv(world_to_camera_with_pose(view_pose))

def transformation(traj_index, first_frame, second_frame):
    file = open(abs_path+str(traj_index)+".txt",'r')
    
    first_camera_x = first_camera_y = first_camera_z = 6.0
    first_lookat_x = first_lookat_y = first_lookat_z = 2.0
    second_camera_x = second_camera_y = second_camera_z = 2.0
    second_lookat_x = second_lookat_y = second_lookat_z = 6.0

    line = file.readline()
    while line:
        if(first_frame == int(line.split(' ')[1])):
            first_lookat_x = float(line.split(' ')[2])
            first_lookat_y = float(line.split(' ')[3])
            first_lookat_z = float(line.split(' ')[4])
            first_camera_x = float(line.split(' ')[5])
            first_camera_y = float(line.split(' ')[6])
            first_camera_z = float(line.split(' ')[7].replace('\n',''))
        
        if(second_frame == int(line.split(' ')[1])):
            second_lookat_x = float(line.split(' ')[2])
            second_lookat_y = float(line.split(' ')[3])
            second_lookat_z = float(line.split(' ')[4])
            second_camera_x = float(line.split(' ')[5])
            second_camera_y = float(line.split(' ')[6])
            second_camera_z = float(line.split(' ')[7].replace('\n',''))

            first_pose = {
                 'lookat':np.array([first_lookat_x,first_lookat_y,first_lookat_z]),
                 'camera':np.array([first_camera_x,first_camera_y,first_camera_z])
            }

            second_pose = {
                 'lookat':np.array([second_lookat_x,second_lookat_y,second_lookat_z]),
                 'camera':np.array([second_camera_x,second_camera_y,second_camera_z])
            }

            r1 = world_to_camera_with_pose(first_pose)
            r2 = world_to_camera_with_pose(second_pose)
            transformation = np.dot(r2, np.linalg.inv(r1))

            file.close()
            return transformation
        
        line = file.readline()
    file.close()


# test = transformation(0,0,50)
# print(test)
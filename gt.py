import numpy as np
import os
#open file
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



with open(os.path.abspath(os.path.dirname(__file__))+"/scenenet/gt.txt",'r') as f:
# read .txt line by line

    #initialization
    frame_num = 0
    so_camera_x=so_camera_y=so_camera_z = 0. #shutter_open camera xyz
    so_lookat_x=so_lookat_y=so_lookat_z = 0. #shutter_open lookat xyz
    sc_camera_x=sc_camera_y=sc_camera_z = 0. #shutter_close camera xyz
    sc_lookat_x=sc_lookat_y=sc_lookat_z = 0. #shutter_close lookat xyz
    so_flag=sc_flag=camere_flag=lookat_flag = 0 #flag for shutter_close and shutter open
    
    # old_pose = {
    #     'lookat': np.array([3,3,4]),
    #     'camera': np.array([3,8,6])
    # }

    line = f.readline() #first line
    while line:
        if('Render path' in line): #create a new txt for new path
            index = line.split('/')[1].replace('\n','')
            file = open(abs_path+index+'.txt','w')
        
        if('frame_num' in line): # get frame number
            frame_num = line.split(':')[1].replace('\n','')

        if('shutter_open' in line): # get shutter_open data
            so_flag = 1
            sc_flag = 0
        
        if('shutter_close' in line): # get shutter_close data
            so_flag = 0
            sc_flag = 1

        if('camera' in line): #get camera data
            camere_flag = 1
            lookat_flag = 0

        if('lookat' in line):
            camere_flag = 0
            lookat_flag = 1
        
        #shutter open camera

        if('x' in line and so_flag and camere_flag):
            so_camera_x = line.split(':')[1].replace('\n','')

        if('y' in line and so_flag and camere_flag):
            so_camera_y = line.split(':')[1].replace('\n','')

        if('z' in line and so_flag and camere_flag):
            so_camera_z = line.split(':')[1].replace('\n','')
        
        #shutter open lookat

        if('x' in line and so_flag and lookat_flag):
            so_lookat_x = line.split(':')[1].replace('\n','')

        if('y' in line and so_flag and lookat_flag):
            so_lookat_y = line.split(':')[1].replace('\n','')

        if('z' in line and so_flag and lookat_flag):
            so_lookat_z = line.split(':')[1].replace('\n','')

        #shutter close camera

        if('x' in line and sc_flag and camere_flag):
            sc_camera_x = line.split(':')[1].replace('\n','')

        if('y' in line and sc_flag and camere_flag):
            sc_camera_y = line.split(':')[1].replace('\n','')

        if('z' in line and sc_flag and camere_flag):
            sc_camera_z = line.split(':')[1].replace('\n','')

        #shutter close lookat

        if('x' in line and sc_flag and lookat_flag):
            sc_lookat_x = line.split(':')[1].replace('\n','')

        if('y' in line and sc_flag and lookat_flag):
            sc_lookat_y = line.split(':')[1].replace('\n','')

        if('z' in line and sc_flag and lookat_flag):
            sc_lookat_z = line.split(':')[1].replace('\n','')

        #write data
        if('timestamp' in line and sc_flag):

            camera_x = (float(sc_camera_x) + float(so_camera_x))/2
            camera_y = (float(sc_camera_y) + float(so_camera_y))/2
            camera_z = (float(sc_camera_z) + float(so_camera_z))/2
            
            lookat_x = (float(sc_lookat_x) + float(so_lookat_x))/2
            lookat_y = (float(sc_lookat_y) + float(so_lookat_y))/2
            lookat_z = (float(sc_lookat_z) + float(so_lookat_z))/2

            # new_pose = {
            #     'lookat':np.array([lookat_x,lookat_y,lookat_z]),
            #     'camera':np.array([camera_x,camera_y,camera_z])
            # }

            file.write(frame_num+' '+str(lookat_x)+' '+str(lookat_y)+' '+str(lookat_z)+' '+str(camera_x)+' '+str(camera_y)+' '+str(camera_z)+'\n')

            
            # r1 = world_to_camera_with_pose(old_pose)
            # r2 = world_to_camera_with_pose(new_pose)
            # transformation = np.dot(r2, np.linalg.inv(r1))

            # file.write(frame_num+' '+str(transformation[0][0])+' '+str(transformation[0][1])+' '+str(transformation[0][2])+' '+str(transformation[0][3])
            #                     +' '+str(transformation[1][0])+' '+str(transformation[1][1])+' '+str(transformation[1][2])+' '+str(transformation[1][3])
            #                     +' '+str(transformation[2][0])+' '+str(transformation[2][1])+' '+str(transformation[2][2])+' '+str(transformation[2][3])
            #                     +' '+str(transformation[3][0])+' '+str(transformation[3][1])+' '+str(transformation[3][2])+' '+str(transformation[3][3])+'\n')

            # old_pose = new_pose
            
        line = f.readline() #read next line
    

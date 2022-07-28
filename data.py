from matplotlib.pyplot import axis
import numpy as np
from torch.utils.data import Dataset
import os
import open3d as o3d
import sys
sys.path.append("dcp-master")
from transformation import transformation
from scipy.spatial.transform import Rotation
import warnings

def load_data(partition, size, interval):
    # download()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'val/0/')
    all_data = []
    scenes = []
    dataset = {'data': all_data,
                'scenes': scenes}
    if partition == 'train':
        for scene in range(121): 
            try:
                all_data = []
                scenes = []
                for image in range(300-interval): # 7475 images
                    data = []
                    data1 = o3d.io.read_point_cloud(DATA_DIR+str(scene)+'/'+str(image*25)+"_after_filtering.pcd")
                    data2 = o3d.io.read_point_cloud(DATA_DIR+str(scene)+'/'+str((image+interval)*25)+"_after_filtering.pcd")
                    index1 = np.random.randint(len(data1.points), size=size)
                    index2 = np.random.randint(len(data2.points), size=size)
                    data1 = np.array(data1.points)[index1]
                    data2 = np.array(data2.points)[index2]
                    data.append(data1)
                    data.append(data2)
                    data = np.array(data)
                    all_data.append(data)
                    scenes.append(scene)
                dataset['data'].append(all_data)
                dataset['scenes'].append(scenes)
            except (Warning, ValueError):
                continue

    if partition == 'val':
        for scene in range(121,151): 
            try:
                all_data = []
                scenes = []
                for image in range(300-interval): # 7475 images
                    data = []
                    data1 = o3d.io.read_point_cloud(DATA_DIR+str(scene)+'/'+str(image*25)+"_after_filtering.pcd")
                    data2 = o3d.io.read_point_cloud(DATA_DIR+str(scene)+'/'+str((image+interval)*25)+"_after_filtering.pcd")
                    index1 = np.random.randint(len(data1.points), size=size)
                    index2 = np.random.randint(len(data2.points), size=size)
                    data1 = np.array(data1.points)[index1]
                    data2 = np.array(data2.points)[index2]
                    data.append(data1)
                    data.append(data2)
                    data = np.array(data)
                    all_data.append(data)
                    scenes.append(scene)
                dataset['data'].append(all_data)
                dataset['scenes'].append(scenes)
            except (Warning, ValueError):
                continue

    dataset['data'] = np.array(dataset['data'])
    dataset['scenes'] = np.array(dataset['scenes'])
    dataset['data'] = np.concatenate(dataset['data'], axis=0)
    dataset['scenes'] = np.concatenate(dataset['scenes'], axis=0)
    return dataset


class SceneNet(Dataset):
    def __init__(self, size, partition, icp=False, interval=1, r=None, t=None):
        self.data = load_data(partition, size, interval)
        self.interval = interval
        self.icp = icp
        self.r = r
        self.t = t

    def __getitem__(self, index):
        if self.icp:
            points1 = self.data['data'][index][0]
            points2 = transform(points1, self.r[index], self.t[index])
        else:
            points1 = self.data['data'][index][0]
            points2 = self.data['data'][index][1]
        T = transformation(self.data['scenes'][index], (index%299)*25, (index%299)*25+self.interval*25)
        R_ab = T[:3,:3]
        R_ba = R_ab.T
        translation_ab = T[:3,3]
        translation_ba = -R_ba.dot(translation_ab)


        euler_ab = npmat2euler(R_ab)
        euler_ba = npmat2euler(R_ba)
        
        return points1.T.astype('float32'), points2.T.astype('float32'), R_ab.astype('float32'), \
        translation_ab.astype('float32'), R_ba.astype('float32'), translation_ba.astype('float32'), \
        euler_ab.astype('float32'), euler_ba.astype('float32')

    def __len__(self):
            return self.data['data'].shape[0]

def npmat2euler(mats, seq='zyx'):
    eulers = []
    r = Rotation.from_matrix(mats)
    eulers.append(r.as_euler(seq, degrees=True))
    return np.asarray(eulers, dtype='float32')

def transform(point_cloud, R=None, t=None):
    t_broadcast = np.broadcast_to(t[:, np.newaxis], (3, point_cloud.shape[0]))
    return (R @ point_cloud.T + t_broadcast).T
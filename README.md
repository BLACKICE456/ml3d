# ml3d final project
## Indoor pictures in rigid transformation using Deep Closest Point (DCP) Initializor
We segment the objects in the images using a dataset, and we use DCP (initialize) and ICP to solve the registration of the objects in order to estimate the pose of the camera.

## Prerequisites
- Python <= 3.9
## Requirements
- OpenCV
- Pytorch
- Numpy
- Go-icp (just for comparing)
- tensorboardX
## Directory
```
.
├── PointNetLK-master/
├── scenenet
│   ├── val/0/
│   └── ground_truth/
├── README.md
├── __init__.py
├── data.py
├── dcp-master
│   ├── data/
│   ├── data.py
│   ├── dcp_matrix.txt
│   ├── main.py
│   ├── model.py
│   ├── pretrained/
│   ├── readme.md
│   └── util.py
├── dcp.py
├── dcp_icp.ipynb
├── dcp_v2.t7
├── filtering/
├── go-icp_cython-master/
├── gt.py
├── icp.py
├── pcd_generator/
├── pcd_process
│   ├── README.md
│   └── pcdtopointcloud.py
├── requirements.txt
├── simpleICP-master
│   ├── LICENSE
│   ├── README.md
│   ├── __init__.py
│   ├── data/
│   ├── icp_matrix.txt
│   ├── simpleicp
│   │   ├── __init__.py
│   │   ├── corrpts.py
│   │   ├── mathutils.py
│   │   ├── optimization.py
│   │   ├── pointcloud.py
│   │   ├── simpleicp.py
│   │   └── tests/
│   ├── test.ipynb
│   └── tests/
├── test.ipynb
├── test_case/
├── transformation.py
└── tree.txt
```
- Using ICP, Go-ICP and PointNetLK for comparing with DCP.  
- Directory **scenenet/** contains all the dataset and its ground truth datas.
## Whole process
1. Considering generating pcd files from the dataset, using **pcd_generator**.
2. Considering generating the ground truth of the dataset using **gt.py**.
3. Considering moving out all the walls and floors of the dataset, using **filtering**.
4. Run **dcp_icp.ipynb** file for the whole training and testing process.
5. Don't forget to modify the number of scenes and the data(with filtering or not) that you want to use in the **data.py**.

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
├── dcp-master/
├── dcp.py
├── dcp_icp.ipynb
├── dcp_v2.t7
├── filtering/
├── go-icp_cython-master/
├── gt.py
├── icp.py
├── pcd_generator/
├── pcd_process/
├── requirements.txt
├── simpleICP-master/
├── test.ipynb
├── test_case/
├── transformation.py
└── tree.txt
```
- Using ICP, Go-ICP and PointNetLK for comparing with DCP.  
- Directory **scenenet/** contains all the dataset and its ground truth datas.
## Whole process
1. Considering generating pcd files from the dataset, using **pcd_generator**.
```
cmake .
make
./rgbd2pcd
```
2. Considering extract the information of dataset (camera pose) using scenenet/read_protobuf
```
make
python read_prtobuf.py
```
3. Considering generating the ground truth of the dataset using **gt.py**.
```python
python gt.py
```
4. Considering moving out all the walls and floors of the dataset, using **filtering**.
```
cmake .
make
./extract
```
5. Run **dcp_icp.ipynb** file for the whole training and testing process.
6. Don't forget to modify the number of scenes and the data(with filtering or not) that you want to use in the **data.py**.

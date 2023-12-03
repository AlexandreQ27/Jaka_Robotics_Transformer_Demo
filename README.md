# Jaka_Demo
This is a grasping project based on jaka robotic arm. We use ROS1 and OpenCV(HSV or GRCNN) to grasp (eye out of hand).The project includes calculating hand-eye matrices, object recognition, and grasping codes include rt-x test.
<img src="grasp.png" alt="jaka">
<img src="rt-x.png" alt="rt-x">
## 实验思路
-1.使用传统抓取方法获得抓取数据

-2.使用robotics-transformer进行自定义数据集训练实现抓取
## General visual grasp(传统方法抓取)(HSV or GRCNN)
### 1.Create a ros workspace
Create workspace：
```bash
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws/src
catkin_init_workspace
```
Compile workspace：
```bash
cd ~/catkin_ws/
catkin_make
```
Set environment variables：
```bash
 echo "source ~/catkin_ws/devel/setup.bash" >> ~/.bashrc
```
Make the above configuration take effect on the current terminal：
```bash
source ~/.bashrc
```
### 3.Perform your own hand-eye calibration（进行自己的手眼标定）
#### Input
- image with chessboard
- robot pose in txt file (xyz(mm),rpy(deg))

#### Output
- camera intrinsic parameters
- eye hand transformations
  
more detail: https://github.com/ZiqiChai/simplified_eye_hand_calibration

### 4.Run
```bash
   roslaunch jaka_driver my_demo.launch
```

## Acknowledgement
reference code: https://github.com/ZiqiChai/simplified_eye_hand_calibration

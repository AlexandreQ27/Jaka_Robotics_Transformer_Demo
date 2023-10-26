#!/usr/bin/python3
import rospy
import cv2
import numpy as np
import pyrealsense2 as rs
from geometry_msgs.msg import Point
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from jaka_msgs.srv import GetObjPos,GetObjPosResponse
data = [ 0.996836 ,-0.0550892 ,-0.0572993 ,  30.01485,
 0.0143201 ,  -0.58461  , 0.811188  , -938.106,
-0.0781854 , -0.809442 , -0.581972   , 459.888,
         0  ,        0     ,     0      ,    1
]  
#-9.50506
#data = [  0.997115, -0.0552664 ,-0.0520413 ,  25.01485,
# 0.0100422 , -0.583489  , 0.812059,   -931.128,
#-0.0752451 , -0.810238  , -0.58125 ,   462.138,
#         0   ,       0  ,        0    ,      1
#]
camera_matrix=np.array([[607.2513784473566, 0, 312.0057806991867],
 [0, 607.5275453454146, 238.6361143847473],
 [0, 0, 1]])
dist_coeff=np.float32([-0.02220346593665167,
 1.294554632487039,
 0.006218227086377285,
 -0.009092912927694383,
 -4.902303897433339])
 
 # 已经有的相机矩阵和畸变系数  
#camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])  
#dist_coefs = np.array([k1, k2, p1, p2, k3])  
#axis: -0.118479   0.93741  0.327451

object_position = Point()
# 将数据转换为NumPy数组并塑造为4x4矩阵  
matrix = np.array(data).reshape((4, 4)) 



ball_color = 'red'
color_dist = {'red': {'Lower': np.array([0, 60, 60]), 'Upper': np.array([10, 255, 255])},
              'blue': {'Lower': np.array([100, 80, 46]), 'Upper': np.array([124, 255, 255])},
              'green': {'Lower': np.array([35, 50, 50]), 'Upper': np.array([80, 255, 255])},
              }
# 创建一个相机实例
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)

# 获取深度传感器的内参
profile = pipeline.get_active_profile()
depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
depth_intrinsics = depth_profile.get_intrinsics()
print(depth_intrinsics)

def object_service_callback(request):
    response=GetObjPosResponse()
    if request.get == True:
        response.x = object_position.x  
        response.y = object_position.y 
        response.z = object_position.z 
        response.message = "Get obj pos has been executed";
    return response
    
# 创建ROS节点和发布器
rospy.init_node('object_detection_node')
#object_pub = rospy.Publisher('/object_position', Point, queue_size=1)
object_service = rospy.Service('/get_object_position',GetObjPos,object_service_callback)  
bridge = CvBridge()

def pixel_to_camera(pixel_x, pixel_y, depth_value):
    # 将像素坐标转换为相机坐标
    camera_point = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [pixel_x, pixel_y], depth_value)
    return camera_point

def main():
    while not rospy.is_shutdown():
        # 获取深度相机的帧
        frames = pipeline.wait_for_frames()

        # 获取彩色图像帧
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # 将图像帧转换为OpenCV格式
        distorted_image = np.asanyarray(color_frame.get_data())
        #camera_matrix_umat = cv2.UMat(camera_matrix)
        #dist_coeff_umat = cv2.UMat(dist_coeff)
        color_image = cv2.undistort(distorted_image, camera_matrix, dist_coeff)
        #color_image=distorted_image

        # 对图像进行处理
        gs_frame = cv2.GaussianBlur(color_image, (5, 5), 0)
        hsv = cv2.cvtColor(gs_frame, cv2.COLOR_BGR2HSV)
        erode_hsv = cv2.erode(hsv, None, iterations=2)
        inRange_hsv = cv2.inRange(erode_hsv, color_dist[ball_color]['Lower'], color_dist[ball_color]['Upper'])
        cv2.imshow('inRange_hsv', inRange_hsv)
        # 开操作
        #kernel = np.ones((5, 5), np.uint8)
        #opened_hsv = cv2.morphologyEx(inRange_hsv, cv2.MORPH_OPEN, kernel)
        # 闭操作
        #closed_hsv = cv2.morphologyEx(opened_hsv, cv2.MORPH_CLOSE, kernel)
        #cv2.imshow('closed_hsv', closed_hsv)
        cnts = cv2.findContours(inRange_hsv.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        if len(cnts) > 0:
            c = max(cnts, key=cv2.contourArea)
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            cv2.drawContours(color_image, [np.int0(box)], -1, (0, 255, 255), 2)
            # 计算物体轮廓的中点坐标
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            	# 在图像上绘制中点坐标
                cv2.circle(color_image, (cx, cy), 5, (0, 255, 0), -1)
            	# 获取深度帧
                depth_frame = frames.get_depth_frame()
                depth_value = depth_frame.get_distance(cx, cy)
            	# 将中点坐标转换为相机坐标系下的坐标
                camera_point = pixel_to_camera(cx, cy, depth_value)
                x_eye = camera_point[0]*1000
                y_eye = camera_point[1]*1000
                z_eye = camera_point[2]*1000
                # 目标在眼坐标系下的坐标 [x_eye, y_eye, z_eye]  
                target_eye_coordinates = np.array([x_eye, y_eye, z_eye, 1]).reshape(4, 1)
                # 计算手上的坐标  
                hand_coordinates = np.dot(matrix, target_eye_coordinates)  
                # 发布物体在相机坐标系下的坐标
                object_position.x = hand_coordinates[0]
                object_position.y = hand_coordinates[1]
                object_position.z = hand_coordinates[2]
                print('{}:{}'.format('x', object_position.x))
                print('{}:{}'.format('y', object_position.y))
                print('{}:{}'.format('z', object_position.z))
                #object_pub.publish(object_position)
        
        cv2.imshow('camera', color_image)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()

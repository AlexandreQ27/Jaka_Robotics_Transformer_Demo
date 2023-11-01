#!/home/qyb/GRCNN/venv/bin/python3
import rospy
import cv2
import numpy as np
import torch
from geometry_msgs.msg import Point
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from jaka_msgs.srv import GetObjPos,GetObjPosResponse
import matplotlib.pyplot as plt
from hardware.camera import RealSenseCamera
from hardware.device import get_device
from inference.post_process import post_process_output
from utils.data.camera_data import CameraData
from utils.dataset_processing.grasp import detect_grasps
from sklearn.preprocessing import MinMaxScaler

saved_model_path='/home/qyb/jaka_robot_v2.2/src/jaka_driver/scripts/saved_data/cornell_rgbd_iou_0.96'
cam_id=213522070067
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


def object_service_callback(request):
    response=GetObjPosResponse()
    if request.get == True:
        response.x = object_position.x  
        response.y = object_position.y 
        response.z = object_position.z 
        response.message = "Get obj pos has been executed";
    return response
    

# def pixel_to_camera(pixel_x, pixel_y, depth_value):
#     # 将像素坐标转换为相机坐标
#     camera_point = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [pixel_x, pixel_y], depth_value)
#     return camera_point

def main():
        # 创建ROS节点和发布器
        rospy.init_node('object_detection_node')
        #object_pub = rospy.Publisher('/object_position', Point, queue_size=1)
        object_service = rospy.Service('/get_object_position',GetObjPos,object_service_callback)  
        bridge = CvBridge()
        camera = RealSenseCamera(device_id=cam_id)
        cam_data = CameraData(include_depth=True, include_rgb=True)

        # Connect to camera
        camera.connect()
        print('Loading model... ')
        model = torch.load(saved_model_path,map_location=torch.device('cpu'))
        while True:
             # Get the compute device
            device = get_device(force_cpu=False)
            # # 将中点坐标转换为相机坐标系下的坐标
            # camera_point = pixel_to_camera(cx, cy, depth_value)
            # Get RGB-D image from camera
            image_bundle = camera.get_image_bundle()
            rgb = image_bundle['rgb']
            depth = image_bundle['aligned_depth']
            x, depth_img, rgb_img = cam_data.get_data(rgb=rgb, depth=depth)

            # Predict the grasp pose using the saved model
            with torch.no_grad():
                xc = x.to(device)
                pred = model.predict(xc)

            q_img, ang_img, width_img = post_process_output(pred['pos'], pred['cos'], pred['sin'], pred['width'])
            grasps = detect_grasps(q_img, ang_img, width_img)
            print(rgb_img.shape)
            rgb_img = np.transpose(rgb_img, (1, 2, 0))
            rgb_img_copy=cv2.resize(rgb, (224, 224))
            # rgb_img_copy=rgb_img.copy()
            # rgb_img_copy=cv2.normalize(rgb_img, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
            #rgb_img_copy = MinMaxScaler(feature_range=(0, 255)).fit_transform(rgb_img)
            print(rgb_img_copy)
            for g in grasps:
                # point_1=g.as_gr.points[0]
                # point_2=g.as_gr.points[1]
                # point_3=g.as_gr.points[2]
                # point_4=g.as_gr.points[3]
                #print(g.center)
                #pt1=int([g.as_gr.points[0][0],g.as_gr.points[0][1]]),pt2=int([g.as_gr.points[1][0],g.as_gr.points[1][1]])
                #print(cam_data.top_left)
                draw_0 = cv2.rectangle(rgb_img_copy, pt1=[int(g.as_gr.points[4][0]),int(g.as_gr.points[4][1])],pt2=[int(g.as_gr.points[5][0]),int(g.as_gr.points[5][1])], color=(255, 0, 0), thickness=2)
                pos_z = depth[grasps[0].center[0] + cam_data.top_left[0], grasps[0].center[1] + cam_data.top_left[1]] 
                pos_x = np.multiply(grasps[0].center[1] + cam_data.top_left[1] - camera.intrinsics.ppx,
                            pos_z / camera.intrinsics.fx)
                pos_y = np.multiply(grasps[0].center[0] + cam_data.top_left[0] - camera.intrinsics.ppy,
                            pos_z / camera.intrinsics.fy)
                target = np.asarray([pos_x, pos_y, pos_z])
                target.shape = (3, 1)
                print('target: ', target)
                x_eye = pos_x*1000
                y_eye = pos_y*1000
                z_eye = pos_z*1000
                # 目标在眼坐标系下的坐标 [x_eye, y_eye, z_eye]  
                target_eye_coordinates = np.array([x_eye[0], y_eye[0], z_eye[0], 1]).reshape(4, 1)
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
            
                cv2.imshow('camera', draw_0)
                if cv2.waitKey(1) & 0xFF == 27 :
                    break 


if __name__ == '__main__':
    main()

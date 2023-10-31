#!/usr/bin/python3
import rospy  
from sensor_msgs.msg import Image, CameraInfo  
from std_msgs.msg import Header  
from message_filters import ApproximateTimeSynchronizer, Subscriber  
from geometry_msgs.msg import PoseStamped  
import cv2  
import yaml  
import os  
from datetime import datetime  
from cv_bridge import CvBridge  

# 定义保存彩色图像的函数  
def save_image(msg, filename):  
    try:  
        cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")  
        cv2.imwrite(filename, cv_image)  
    except CvBridgeError as e:  
        print(e)  

def save_pose_stamped(msg, filename):  
    # 提取x, y, z数据  
    x = msg.pose.position.x  
    y = msg.pose.position.y  
    z = msg.pose.position.z  
    rx = msg.pose.orientation.x  
    ry = msg.pose.orientation.y  
    rz = msg.pose.orientation.z  
    # 将数据写入文本文件  
    with open(filename, 'a') as f:  
        f.write(f'{x} {y} {z} {rx} {ry} {rz}\n')
        
         

# 定义回调函数来处理彩色图像和PoseStamped消息   
def color_callback(msg):  
    global has_saved_img, color_dir, bridge  
    stamp = rospy.Time.now() # 你可能需要使用你自己的时间戳，这里只是一个例子。 
    # 获取文件夹中已有的文件列表  
    files = os.listdir(color_dir)  
    # 过滤出以 '.png' 结尾的文件，并按文件名中的数字排序  
    png_files = sorted([file for file in files if file.endswith('.png')], key=lambda x: int(x.split('.')[0]))  
    # 找到最大的文件编号  
    max_file_number = int(png_files[-1].split('.')[0]) if png_files else -1 
    # 新的文件编号  
    new_file_number = max_file_number + 1   
    color_filename = color_dir + '{}.png'.format(new_file_number) 
    if not has_saved_img: 
        save_image(msg, color_filename)  
        print("img have saved")
        has_saved_img = True  
    
    
 
def pose_callback(msg):  
    global has_saved_pose, bridge, pose_sub  
    stamp = rospy.Time.now() # 你可能需要使用你自己的时间戳，这里只是一个例子。  
    pose_filename = color_dir + 'pose.txt'
    if not has_saved_pose:
        save_pose_stamped(msg, pose_filename)  
        print("pose have saved")
        has_saved_pose = True    
# 主函数  
if __name__ == '__main__':  
    rospy.init_node('data_saver')  
    # 初始化 CvBridge 对象  
    bridge = CvBridge()  
    has_saved_img=False
    has_saved_pose=False
    # 定义保存彩色图像和深度图像的目录和文件名  
    color_dir = 'color_images/'  
    if not os.path.exists(color_dir):  
        os.makedirs(color_dir)  
  
    # 创建三个订阅器，分别订阅相机彩色图像、深度图像和PoseStamped话题  
    color_sub = rospy.Subscriber('/camera/color/image_raw', Image,color_callback)  
    pose_sub = rospy.Subscriber('/jaka_driver/tool_point', PoseStamped,pose_callback)  
    #rate = rospy.Rate(10) # 10hz  应该根据你的需求来设置这个频率。如果你的订阅器已经注册了回调函数，你可能不需要这个循环。
    rospy.spin() 
    

#!/home/qyb/RT-1/rt-1/bin/python3
import rospy, math, random, cv_bridge, cv2
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped  
from jaka_msgs.msg import IOMsg
from cv_bridge import CvBridge, CvBridgeError
import message_filters
import os
# 定义全局变量
bridge = CvBridge()
save_dir_template = "~/data/episode_{}"
current_episode_number = 0


def get_next_save_dir():
    global current_episode_number
    saving_dir = save_dir_template.format(current_episode_number)
    while os.path.exists(os.path.expanduser(saving_dir)):  # 添加os.path.expanduser处理~
        current_episode_number += 1
        saving_dir = save_dir_template.format(current_episode_number)
    
    # 创建目录，如果它还不存在
    os.makedirs(os.path.expanduser(saving_dir), exist_ok=True)
    # pose_filename = os.path.join(os.path.expanduser(saving_dir), "pose.txt")

    # 创建文件，如果它还不存在
    # with open(pose_filename, 'a'): pass  # 创建空文件
    return os.path.expanduser(saving_dir)  # 返回最新生成的saving_dir
    

save_dir = get_next_save_dir()



print(f"The next directory is: {save_dir}")


def save_image(msg, filename):  
    try:  
        cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")  
        cv2.imwrite(filename, cv_image)  
    except CvBridgeError as e:  
        print(e)  

def save_pose_stamped(tool_point_msg, gripper_msg,filename):   
    x = tool_point_msg.pose.position.x  
    y = tool_point_msg.pose.position.y  
    z = tool_point_msg.pose.position.z  
    rx = tool_point_msg.pose.orientation.x  
    ry = tool_point_msg.pose.orientation.y  
    rz = tool_point_msg.pose.orientation.z  
    gripper = gripper_msg.io_state
    # 将数据写入文本文件  
    with open(filename, 'a') as f:  
        f.write(f'{x} {y} {z} {rx} {ry} {rz} {gripper}\n')



def multi_callback(rgb_image_msg, tool_point_msg,gripper_msg):
    global save_dir
    print("同步start！")
    try:
        color = bridge.imgmsg_to_cv2(rgb_image_msg, desired_encoding='bgr8')
        # depth = bridge.imgmsg_to_cv2(depth_image_msg, desired_encoding='8UC1') 
        files = os.listdir(os.path.expanduser(save_dir))  
        # 过滤出以 '.png' 结尾的文件，并按文件名中的数字排序,找到最大的文件编号 
        png_files = sorted([file for file in files if file.endswith('.png')], key=lambda x: int(x.split('.')[0]))   
        max_file_number = int(png_files[-1].split('.')[0]) if png_files else -1 
        new_file_number = max_file_number + 1   
        color_filename = save_dir + '/{}.png'.format(new_file_number) 
        save_image(rgb_image_msg, color_filename)  
        print("img have saved")
        pose_filename = save_dir + '/pose.txt'
        save_pose_stamped(tool_point_msg, gripper_msg, pose_filename)  
        print("pose have saved")
        

    except CvBridgeError as e:
        rospy.logerr(f"Error converting image: {e}")

    print("同步完成！")

if __name__ == '__main__':
    rospy.init_node('sync_topics', anonymous=True)

    subcriber_rgb = message_filters.Subscriber('/new_color_image_raw', Image)
    subcriber_pose = message_filters.Subscriber('/jaka_driver/tool_point', PoseStamped)  
    subcriber_gripper = message_filters.Subscriber("/jaka_driver/gripper", IOMsg);
    # subcriber_gripper = message_filters.Subscriber()
    print("订阅成功!")

    sync = message_filters.ApproximateTimeSynchronizer([subcriber_rgb, subcriber_pose,subcriber_gripper], 2, 1)
    sync.registerCallback(multi_callback)
    print("同步器初始化完成!")

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("\nClosing...")


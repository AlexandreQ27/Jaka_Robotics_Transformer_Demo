#!/home/qyb/RT-1/rt-1/bin/python3
import rospy, math, random, cv_bridge, cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import message_filters
def rgb_callback(image):
     rospy.loginfo("got RGB image")

def depth_callback(image):
    rospy.loginfo("got DEPTH image")

def scan_callback(image,depth):
    rospy.loginfo("got synched images")


def hand_tracker():

    rospy.init_node('hand_tracker')
    subcriber_rgb = message_filters.Subscriber('/camera/rgb/image_raw', Image, queue_size=10)
    subcriber_depth = message_filters.Subscriber('/camera/depth/image_rect_raw', Image, queue_size=10)   

           
    ts=message_filters.ApproximateTimeSynchronizer([subcriber_rgb,subcriber_depth],10,1)    
    ts.registerCallback(scan_callback)
    try:
        rospy.spin()
    except KeyboardInterrupt:        
        print("Shutting down")       

if __name__ == '__main__':
    hand_tracker()

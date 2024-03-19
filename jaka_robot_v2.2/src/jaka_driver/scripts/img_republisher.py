#!/home/qyb/RT-1/rt-1/bin/python3
import rospy
from sensor_msgs.msg import Image
import message_filters


def callback(image_msg):
    # 如果时间戳为空，这里可以设置当前时间戳
    if not image_msg.header.stamp:
        image_msg.header.stamp = rospy.Time.now()
    
    new_image_publisher.publish(image_msg)


if __name__ == '__main__':
    try:
        rospy.init_node('image_republisher_node', anonymous=True)
        # 创建一个新的Publisher，用于发布带有时间戳的图像
        new_image_publisher = rospy.Publisher('/new_color_image_raw', Image, queue_size=10)
        # 订阅原始图像话题
        image_subscriber = rospy.Subscriber('/camera/color/image_raw', Image, callback)
        rospy.spin()  # 保持节点运行直到关闭
    except rospy.ROSInterruptException:
        pass

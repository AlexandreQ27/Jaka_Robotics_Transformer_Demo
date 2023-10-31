#!/usr/bin/python3
import rospy
import time
import sys
sys.path.append('/home/qyb/jaka_robot_v2.2/src/jaka_driver/lib_python/')
import jkrc

if __name__ == '__main__':
    rospy.init_node('my_node')   # 初始化ROS节点
    robot = jkrc.RC("192.168.1.100")  # 创建机器人对象

    rospy.sleep(0.5)  # 等待ROS节点初始化完成

    # 登录
    ret = robot.login()
    if ret[0] == 0:
        rospy.loginfo("Login successful")
    else:
        rospy.logerr("Login failed, error code: %s", ret[0])

    # 查询工具坐标系数据
    ret = robot.get_tool_data(0)
    if ret[0] == 0:
        rospy.loginfo("The tool data is: %s", ret)
    else:
        rospy.logerr("Something happened, error code: %s", ret[0])
    # 再次查询工具坐标系数据
    ret = robot.get_tool_data(1)
    if ret[0] == 0:
        rospy.loginfo("The tool data is: %s", ret)
    else:
        rospy.logerr("Something happened, error code: %s", ret[0])

    # 查询工具坐标系id
    ret = robot.get_tool_id()
    rospy.loginfo("Tool ID: %s", ret)
    
    # 登出
    robot.logout()
    rospy.loginfo("Logout")

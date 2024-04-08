from sensor_msgs import JointState
import rospy
import jkrc
ABS=0
INCR=1
speed=1
def client_move_callback(msg):
    joint_pos=msg.position
    speed = 50.0
    accel = 50.0
    ret = robot.joint_move(joint_pos, ABS, True, speed)
    rospy.spin()

# 主函数  
if __name__ == '__main__':  
    rospy.init_node('synchronization')  
    robot = jkrc.RC("192.168.2.160")
    robot.login()#登录
    robot.power_on() #上电
    robot.enable_robot()
    pose_sub = rospy.Subscriber('/jaka_driver/tool_point', JointState,client_move_callback) 
    while(ros::ok())
    {   
        // tool_position_callback(tool_position_pub);
        //tool_point_callback(tool_point_pub);
        // robot_states_callback(robot_state_pub);
        rate.sleep();
        ros::spinOnce();
    }
    rospy.spin()


#include "ros/ros.h"
#include "std_msgs/String.h"
#include "std_msgs/Float32.h"
#include "std_srvs/Empty.h"
#include "std_srvs/SetBool.h"
#include "geometry_msgs/TwistStamped.h"
#include "sensor_msgs/JointState.h"
#include "geometry_msgs/PoseStamped.h"
#include "Eigen/Dense"
#include "Eigen/Core"
#include "Eigen/Geometry"
#include "Eigen/StdVector"

#include "jaka_msgs/RobotMsg.h"
#include "jaka_msgs/IOMsg.h"
#include "jaka_msgs/Move.h"
#include "jaka_msgs/ServoMoveEnable.h"
#include "jaka_msgs/ServoMove.h"
#include "jaka_msgs/SetUserFrame.h"
#include "jaka_msgs/SetTcpFrame.h"
#include "jaka_msgs/SetPayload.h"
#include "jaka_msgs/SetCollision.h"
#include "jaka_msgs/SetIO.h"
#include "jaka_msgs/GetIO.h"
#include "jaka_msgs/GetFK.h"
#include "jaka_msgs/GetIK.h"
#include "jaka_msgs/ClearError.h"

#include "jaka_driver/JAKAZuRobot.h"
#include "jaka_driver/jkerr.h"
#include "jaka_driver/jktypes.h"
#include "jaka_driver/conversion.h"

#include <actionlib/server/simple_action_server.h>
#include <control_msgs/FollowJointTrajectoryAction.h>
#include <trajectory_msgs/JointTrajectory.h>

#include <string>
#include <map>
#include <chrono>
#include <thread>

using namespace std;
const double PI = 3.1415926;
BOOL in_pos;
JAKAZuRobot robot;

//SDK interface return status
map<int, string>mapErr = {
    {2,"ERR_FUCTION_CALL_ERROR"},
    {-1,"ERR_INVALID_HANDLER"},
    {-2,"ERR_INVALID_PARAMETER"},
    {-3,"ERR_COMMUNICATION_ERR"},
    {-4,"ERR_KINE_INVERSE_ERR"},
    {-5,"ERR_EMERGENCY_PRESSED"},
    {-6,"ERR_NOT_POWERED"},
    {-7,"ERR_NOT_ENABLED"},
    {-8,"ERR_DISABLE_SERVOMODE"},
    {-9,"ERR_NOT_OFF_ENABLE"},
    {-10,"ERR_PROGRAM_IS_RUNNING"},
    {-11,"ERR_CANNOT_OPEN_FILE"},
    {-12,"ERR_MOTION_ABNORMAL"}
};

const double tolerance = 3.0;
double average_x;
double average_y;
double average_z;

double cur_pos_x;
double cur_pos_y;
double cur_pos_z;


ros::Publisher joint_position_pub ;

//author:QIU
void joint_position_callback(ros::Publisher joint_position_pub)
{
    sensor_msgs::JointState joint_position;
    RobotStatus robotstatus;
    robot.get_robot_status(&robotstatus);
    for (int i = 0; i < 6; i++)
    {
        joint_position.position.push_back(robotstatus.joint_position[i]);
        int j = i + 1;
        joint_position.name.push_back("joint_" + to_string(j));
    }
    joint_position.header.stamp = ros::Time::now();
    joint_position_pub.publish(joint_position);
}

void* get_conn_scoket_state(void* args){
    RobotStatus robot_status;
	
    while (ros::ok())
    {
        int ret = robot.get_robot_status(&robot_status);
		if (ret)
        {
            ROS_ERROR("get_robot_status error!!!");
        }
        else if(!robot_status.is_socket_connect)
		{
            ROS_ERROR("connect error!!!");
        }
        if(ret==0)
        {
            joint_position_callback(joint_position_pub);
        }
        // ros::Duration(0.1).sleep(); 
    }    
}

int main(int argc, char *argv[])
{
    setlocale(LC_ALL, "");
    ros::init(argc, argv, "synchronization_server");
    ros::NodeHandle nh;
    string default_ip = "192.168.1.101";
    string robot_ip = nh.param("ip_server", default_ip);
    robot.login_in(robot_ip.c_str());
    robot.set_status_data_update_time_interval(100);
    robot.set_block_wait_timeout(120);
    robot.power_on();
    sleep(1);
    robot.enable_robot();
    //Joint-space first-order low-pass filtering in robot servo mode
    //robot.servo_move_use_joint_LPF(2);
    robot.servo_speed_foresight(15,0.03);
     //author:QIU
    ros::Rate rate(125);
    //ros::Subscriber client_move_sub = nh.subscribe("/jaka_driver/joint_position", 1, client_move_callback);
    joint_position_pub = nh.advertise<sensor_msgs::JointState>("/jaka_driver/joint_position", 1);

    pthread_t conn_state_thread;
    int ret = pthread_create(&conn_state_thread,NULL,get_conn_scoket_state,NULL);
    
    while(ros::ok())
    {   
        // cout<<"OK"<<endl;
        // tool_position_callback(tool_position_pub);
        //tool_point_callback(tool_point_pub);
        // robot_states_callback(robot_state_pub);
        // rate.sleep();
        ros::spinOnce();
    }
    // ros::Duration(0.1).sleep();
    return 0;
}

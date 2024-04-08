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



ros::Publisher tool_point_pub ;
ros::Publisher gripper_pub;

//author:QIU
void tool_point_callback(ros::Publisher tool_point_pub)
{
    geometry_msgs::PoseStamped  tool_point;
    CartesianPose tcp_pos;
    robot.get_tcp_position(&tcp_pos);
    tool_point.pose.position.x = tcp_pos.tran.x;
    tool_point.pose.position.y = tcp_pos.tran.y;
    tool_point.pose.position.z = tcp_pos.tran.z;
    
    tool_point.pose.orientation.x = (tcp_pos.rpy.rx )/PI*180;
    tool_point.pose.orientation.y = (tcp_pos.rpy.ry )/PI*180;
    tool_point.pose.orientation.z = (tcp_pos.rpy.rz )/PI*180;
    
    //tool_point.pose.orientation.x = (tcp_pos.rpy.rx );
    //tool_point.pose.orientation.y = (tcp_pos.rpy.ry );
    //tool_point.pose.orientation.z = (tcp_pos.rpy.rz );


    
    tool_point.header.stamp = ros::Time::now();
    tool_point_pub.publish(tool_point);
}


//author:QIU
void gripper_callback(ros::Publisher io_pub)
{   
    jaka_msgs::IOMsg io_data;
    IOType type;
    int ret;
    BOOL digital_result;
    float analog_result;
    type = IO_CABINET;
    float value;
    string signal = "digital";
    int index = 2;
    //ret = robot.get_digital_input(type, index, &digital_result);
    ret = robot.get_digital_output(type, index, &digital_result);
    switch(ret)
    {
        case 0:
            value = float(digital_result);
            break;
        default:
            value = -999999;
            ROS_ERROR("Error occurred: %s", mapErr[ret].c_str());
    }
    io_data.io_state = value;
    io_data.header.stamp = ros::Time::now();
    io_pub.publish(io_data);
    
}


void client_move_callback(const sensor_msgs::JointState &msg)
{
    JointValue joint_pose;
    joint_pose.jVal[0] = msg.position[0];
    joint_pose.jVal[1] = msg.position[1];
    joint_pose.jVal[2] = msg.position[2];
    joint_pose.jVal[3] = msg.position[3];
    joint_pose.jVal[4] = msg.position[4]; 
    joint_pose.jVal[5] = msg.position[5];
    double speed = 100.0;
    double accel = 5.0;
    double tol = 0.5;
    OptionalCond *option_cond = nullptr;

    int ret = robot.joint_move(&joint_pose, MoveMode::ABS, true, speed, accel, tol, option_cond);
    switch(ret)
    {
        case 0:
            ROS_INFO("joint_move has been executed");
    }
    ros::spinOnce();
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
            tool_point_callback(tool_point_pub);
            gripper_callback(gripper_pub);
        }
        // ros::Duration(0.1).sleep(); 
    }    
}

int main(int argc, char *argv[])
{
    setlocale(LC_ALL, "");
    ros::init(argc, argv, "synchronization_client");
    ros::NodeHandle nh;
    string default_ip = "192.168.1.101";
    string robot_ip = nh.param("ip_client", default_ip);
    robot.login_in(robot_ip.c_str());
    robot.set_status_data_update_time_interval(100);
    robot.set_block_wait_timeout(120);
    robot.power_on();
    sleep(1);
    robot.enable_robot();
    //Joint-space first-order low-pass filtering in robot servo mode
    //robot.servo_move_use_joint_LPF(2);
    robot.servo_speed_foresight(15,0.03);
    ros::Rate rate(125);
    ros::Subscriber client_move_sub = nh.subscribe("/jaka_driver/joint_position", 1, client_move_callback);
    //author:QIU
    tool_point_pub = nh.advertise<geometry_msgs::PoseStamped>("/jaka_driver/tool_point", 10);
    gripper_pub = nh.advertise<jaka_msgs::IOMsg>("/jaka_driver/gripper",10);

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

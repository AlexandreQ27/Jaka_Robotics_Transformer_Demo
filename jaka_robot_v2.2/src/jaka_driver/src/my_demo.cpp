#include "ros/ros.h"
#include "std_msgs/String.h"
#include "std_srvs/Empty.h"
#include "std_srvs/SetBool.h"
#include "geometry_msgs/TwistStamped.h"
#include <geometry_msgs/Point.h>
#include "sensor_msgs/JointState.h"
#include "Eigen/Dense"
#include "Eigen/Core"
#include "Eigen/Geometry"
#include "Eigen/StdVector"
#include "jaka_msgs/RobotMsg.h"
#include "jaka_msgs/Move.h"
#include "jaka_msgs/ServoMoveEnable.h"
#include "jaka_msgs/ServoMove.h"
#include "jaka_msgs/SetUserFrame.h"
#include "jaka_msgs/SetTcpFrame.h"
#include "jaka_msgs/SetPayload.h"
#include "jaka_msgs/SetCollision.h"
#include "jaka_msgs/ClearError.h"
#include "jaka_msgs/GetObjPos.h"
#include "jaka_driver/JAKAZuRobot.h"
#include "jaka_driver/jkerr.h"
#include "jaka_driver/jktypes.h"
#include "jaka_driver/conversion.h"
#include <string>
using namespace std;
const double PI = 3.1415926;
BOOL in_pos;
JAKAZuRobot robot;
double average_x;
double average_y;
double average_z;


int main(int argc, char *argv[])
{
    setlocale(LC_ALL, "");
    ros::init(argc, argv, "client_test");
    ros::NodeHandle nh;
    ros::ServiceClient linear_move_client = nh.serviceClient<jaka_msgs::Move>("/jaka_driver/linear_move");
    ros::ServiceClient object_position_client = nh.serviceClient<jaka_msgs::GetObjPos>("/get_object_position");
    linear_move_client.waitForExistence();
    object_position_client.waitForExistence();
    //ros::Subscriber sub = nh.subscribe("/object_position", 1000, object_position_callback); 
    //ros::Duration(1).sleep();
    jaka_msgs::Move linear_move_pose;
    jaka_msgs::GetObjPos obj_pose;
    obj_pose.request.get=true;
    for (int i = 0; i < 10; i++)
    {
        object_position_client.call(obj_pose);
        cout << "The return value of calling obj_pose:" << obj_pose.response.x << "  ";
        cout << obj_pose.response.message << endl;
    }
    float pose[6] = {-200.0, -70.0, 190.0, -2.0, 2.0, 0.0};
    for (int i =0; i < 6; i++)
    {
        linear_move_pose.request.pose.push_back(pose[i]);
    } 
    linear_move_pose.request.has_ref=false;
    linear_move_pose.request.ref_joint={0};
    linear_move_pose.request.mvvelo=10;
    linear_move_pose.request.mvacc=10;
    linear_move_pose.request.mvtime=0.0;
    linear_move_pose.request.mvradii=0.0;
    linear_move_pose.request.coord_mode=0;
    linear_move_pose.request.index=0;
    cout << linear_move_pose.request;
    for (int i = 0; i < 10; i++)
    {
        linear_move_client.call(linear_move_pose);
        cout << "The return value of calling linear_move:" << linear_move_pose.response.ret << "  ";
        cout << linear_move_pose.response.message << endl;
    }
    ros::Duration(1).sleep();
    return 0;
}

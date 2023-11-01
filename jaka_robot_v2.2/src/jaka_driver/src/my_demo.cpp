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
#include "jaka_msgs/SetIO.h"
#include "jaka_driver/JAKAZuRobot.h"
#include "jaka_driver/jkerr.h"
#include "jaka_driver/jktypes.h"
#include "jaka_driver/conversion.h"
#include "geometry_msgs/PoseStamped.h"
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
    ros::ServiceClient set_io_client = nh.serviceClient<jaka_msgs::SetIO>("/jaka_driver/set_io");
    //geometry_msgs::PoseStamped  tool_point;
    //CartesianPose tcp_pos;
    //robot.get_tcp_position(&tcp_pos);
    //cout << tcp_pos.tran.x <<' '<< tcp_pos.tran.y <<' '<< tcp_pos.tran.z<<endl;
    linear_move_client.waitForExistence();
    object_position_client.waitForExistence();
    set_io_client.waitForExistence();
    jaka_msgs::Move linear_move_pose_start,linear_move_pose_end;
    jaka_msgs::GetObjPos obj_pose;
    jaka_msgs::SetIO set_io;
    obj_pose.request.get=true;
    float obj_pos_x,obj_pos_y;
    for(int i=0;i<5;i++)
    {
        for (int i = 0; i < 50; i++)
        {
            object_position_client.call(obj_pose);
            cout << "The return value of calling obj_pose:" << obj_pose.response.x << "  ";
            obj_pos_x=obj_pose.response.x;
            obj_pos_y=obj_pose.response.y;
            cout << obj_pose.response.message << endl;
        }
        cout<<obj_pos_x<<endl;
        cout<<obj_pos_y<<endl;
        int confirm;
        cout<<"confirm end point"<<endl;
        cin>>confirm;
        if(confirm!=1)
            continue;
        float start_pose[6] = {obj_pos_x, obj_pos_y, -0.0, -150.0/180.0*PI, 30.0/180.0*PI, 150.0/180.0*PI};
        //float pose[6] = {111.126,282.111,271.55,3.142,0,-0.698};
        for (int i =0; i < 6; i++)
        {
            linear_move_pose_start.request.pose.push_back(start_pose[i]);
        } 
        linear_move_pose_start.request.has_ref=false;
        linear_move_pose_start.request.ref_joint={0};
        linear_move_pose_start.request.mvvelo=100;
        linear_move_pose_start.request.mvacc=100;
        linear_move_pose_start.request.mvtime=0.0;
        linear_move_pose_start.request.mvradii=0.0;
        linear_move_pose_start.request.coord_mode=0;
        linear_move_pose_start.request.index=0;
        cout << linear_move_pose_start.request;
        for (int i = 0; i < 200; i++)
        {
            linear_move_client.call(linear_move_pose_start);
            cout << "The return value of calling linear_move:" << linear_move_pose_start.response.ret << "  ";
            cout << linear_move_pose_start.response.message << endl;
        }
        set_io.request.signal="digital";
        set_io.request.type=0;
        set_io.request.value=true;
        set_io.request.index=2;
        for (int i = 0; i < 10; i++)
        {
            set_io_client.call(set_io);
            cout << "The return value of calling set_io:"<< set_io.response.message << endl;
        }
        float end_pose[6] = {34.0, -414.0, 200.0, -150.0/180.0*PI, 30.0/180.0*PI, 150.0/180.0*PI};
        //float pose[6] = {111.126,282.111,271.55,3.142,0,-0.698};
        for (int i =0; i < 6; i++)
        {
            linear_move_pose_end.request.pose.push_back(end_pose[i]);
        } 
        linear_move_pose_end.request.has_ref=false;
        linear_move_pose_end.request.ref_joint={0};
        linear_move_pose_end.request.mvvelo=100;
        linear_move_pose_end.request.mvacc=100;
        linear_move_pose_end.request.mvtime=0.0;
        linear_move_pose_end.request.mvradii=0.0;
        linear_move_pose_end.request.coord_mode=0;
        linear_move_pose_end.request.index=0;
        cout << linear_move_pose_end.request;
        for (int i = 0; i < 100; i++)
        {
            linear_move_client.call(linear_move_pose_end);
            cout << "The return value of calling linear_move:" << linear_move_pose_end.response.ret << "  ";
            cout << linear_move_pose_end.response.message << endl;
        }
        set_io.request.signal="digital";
        set_io.request.type=0;
        set_io.request.value=false;
        set_io.request.index=2;
        for (int i = 0; i < 10; i++)
        {
            set_io_client.call(set_io);
            cout << "The return value of calling set_io:"<< set_io.response.message << endl;
        }
    }
    ros::Duration(1).sleep();
    return 0;
}

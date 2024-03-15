#!/home/qyb/RT-1/rt-1/bin/python3
import json
import socket
import time

def send_joint_state_command(command,server_ip):
    command_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    command_socket.connect((server_ip, 10001))
    command_socket.sendall(json.dumps(command).encode())
    response = command_socket.recv(1024).decode()
    response_json = json.loads(response)
    command_socket.close()

    return response_json


def send_joint_move_command(move_params, server_ip):
    command_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    command_socket.connect((server_ip, 10001))
    command_socket.sendall(json.dumps(move_params).encode())
    response = command_socket.recv(1024).decode()
    response_json = json.loads(response)
    command_socket.close()

    # 检查错误码，如果非零则可能需要处理错误
    if int(response_json["errorCode"]) != 0:
        print(f"Joint move error: {response_json['errorMsg']}")

    return response_json
    
    
def main(server_ip,client_ip):
    while True:
        # 获取关节位置数据
        command = {"cmdName": "get_joint_pos"}
        response = send_joint_state_command(command, server_ip=server_ip)
        current_joint_position = response['joint_pos']
        print(f"State Server responded with: {response}")
        incremented_joint_position = current_joint_position.copy()  
        incremented_joint_position[-1] -= 5  # 只增加最后一个关节的角度

        move_params = {
            "cmdName": "joint_move",
            "relFlag": 0,
            "jointPosition": incremented_joint_position,
            "speed": 20.5,
            "accel": 20.5
        }
        response = send_joint_move_command(move_params, server_ip=client_ip)
        print(f"Move Server responded with: {response}")

        # 根据实际情况决定是否在此处暂停
        time.sleep(1)  

if __name__ == "__main__":
    main('192.168.1.101','192.168.1.101')  # 替换为实际的服务器IP地址

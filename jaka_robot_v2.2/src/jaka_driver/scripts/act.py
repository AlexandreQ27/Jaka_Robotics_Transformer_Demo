#!/home/qyb/act/act/bin/python3
import rospy
import cv2
import numpy as np
import pyrealsense2 as rs
from geometry_msgs.msg import Point
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from jaka_msgs.srv import ACTMsg,ACTMsgResponse
from policy import ACTPolicy, CNNMLPPolicy
from utils import set_seed
import torch
import numpy as np
import os
import pickle
import argparse
import IPython
import h5py
from einops import rearrange
e = IPython.embed
import os

def get_args():
    parser = argparse.ArgumentParser(description='获得训练参数')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=True)
    parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize', required=True)
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    parser.add_argument('--batch_size', action='store', type=int, help='batch_size', required=True)
    parser.add_argument('--seed', action='store', type=int, help='seed', required=True)
    parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs', required=True)
    parser.add_argument('--lr', action='store', type=float, help='lr', required=True)

    # for ACT
    parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', required=False)
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', required=False)
    parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', required=False)
    parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward', required=False)
    parser.add_argument('--temporal_agg', action='store_true')
    args = parser.parse_known_args()[0]
    return args


def check_array_condition(arr):
    # 检查第一个元素是否为1
    return bool(arr[0])
    
    
def make_policy(policy_class, policy_config):
    if policy_class == 'ACT':
        policy = ACTPolicy(policy_config)
    elif policy_class == 'CNNMLP':
        policy = CNNMLPPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy


def make_optimizer(policy_class, policy):
    if policy_class == 'ACT':
        optimizer = policy.configure_optimizers()
    elif policy_class == 'CNNMLP':
        optimizer = policy.configure_optimizers()
    else:
        raise NotImplementedError
    return optimizer


# 配置相机
bridge = CvBridge()
target_position = Point()
# 创建一个相机实例
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)

# 获取深度传感器的内参
profile = pipeline.get_active_profile()
depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
depth_intrinsics = depth_profile.get_intrinsics()
print(depth_intrinsics)



# 获取训练脚本中的参数
args = get_args()  # 假设你已经实现了get_args函数    
set_seed(1)
# command line parameters
is_eval = args['eval']
ckpt_dir = args['ckpt_dir']
policy_class = args['policy_class']
onscreen_render = args['onscreen_render']
task_name = args['task_name']
batch_size_train = args['batch_size']
batch_size_val = args['batch_size']
num_epochs = args['num_epochs']

# get task parameters
is_sim = task_name[:4] == 'sim_'

from aloha_scripts.constants import TASK_CONFIGS
task_config = TASK_CONFIGS[task_name]
dataset_dir = task_config['dataset_dir']
num_episodes = task_config['num_episodes']
episode_len = task_config['episode_len']
camera_names = task_config['camera_names']

# fixed parameters
state_dim = 7
lr_backbone = 1e-5
backbone = 'resnet18'
if policy_class == 'ACT':
    enc_layers = 4
    dec_layers = 7
    nheads = 8
    policy_config = {'lr': args['lr'],
                        'num_queries': args['chunk_size'],
                        'kl_weight': args['kl_weight'],
                        'hidden_dim': args['hidden_dim'],
                        'dim_feedforward': args['dim_feedforward'],
                        'lr_backbone': lr_backbone,
                        'backbone': backbone,
                        'enc_layers': enc_layers,
                        'dec_layers': dec_layers,
                        'nheads': nheads,
                        'camera_names': camera_names,
                        }
elif policy_class == 'CNNMLP':
    policy_config = {'lr': args['lr'], 'lr_backbone': lr_backbone, 'backbone' : backbone, 'num_queries': 1,
                        'camera_names': camera_names,}
else:
    raise NotImplementedError

config = {
    'num_epochs': num_epochs,
    'ckpt_dir': ckpt_dir,
    'episode_len': episode_len,
    'state_dim': state_dim,
    'lr': args['lr'],
    'policy_class': policy_class,
    'onscreen_render': onscreen_render,
    'policy_config': policy_config,
    'task_name': task_name,
    'seed': args['seed'],
    'temporal_agg': args['temporal_agg'],
    'camera_names': camera_names,
    'real_robot': not is_sim
}

ckpt_name = [f'policy_best.ckpt']
episode_id = 0

ckpt_dir = config['ckpt_dir']
state_dim = config['state_dim']
real_robot = config['real_robot']
policy_class = config['policy_class']
onscreen_render = config['onscreen_render']
policy_config = config['policy_config']
camera_names = config['camera_names']
max_timesteps = config['episode_len']
task_name = config['task_name']
temporal_agg = config['temporal_agg']
onscreen_cam = 'angle'

# load policy and stats
ckpt_path = os.path.join(ckpt_dir, ckpt_name)
policy = make_policy(policy_class, policy_config)
loading_status = policy.load_state_dict(torch.load(ckpt_path))
print(loading_status)
policy.cuda()
policy.eval()
print(f'Loaded: {ckpt_path}')
stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
with open(stats_path, 'rb') as f:
    stats = pickle.load(f)

pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
post_process = lambda a: a * stats['action_std'] + stats['action_mean']


query_frequency = policy_config['num_queries']
if temporal_agg:
    query_frequency = 1
    num_queries = policy_config['num_queries']

max_timesteps = int(max_timesteps * 1) # may increase for real-world tasks

num_rollouts = 50
episode_returns = []

### evaluation loop
if temporal_agg:
    all_time_actions = torch.zeros([max_timesteps, max_timesteps+num_queries, state_dim]).cuda()

qpos_history = torch.zeros((1, max_timesteps, state_dim)).cuda()
image_list = [] # for visualization
qpos_list = []
target_qpos_list = []


t = 0


#rt-1 model init


def act_service_callback(request):
    frames = pipeline.wait_for_frames()

    # 获取彩色图像帧
    color_frame = frames.get_color_frame()
    if not color_frame:
        return
    # 预处理图像
    color_image = np.asanyarray(color_frame.get_data())
    with torch.inference_mode():
        ### process previous timestep to get qpos and image_list
        # obs = {
        #     'images':all_images[t],
        #     'qpos':all_qpos[t]
        # }
        # if 'images' in obs:
        #     image_list.append(obs['images'])
        # else:
        #     image_list.append({'main': obs['image']})
        qpos_numpy = np.array(obs['qpos'])
        qpos = pre_process(qpos_numpy)
        qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
        qpos_history[:, t] = qpos
        curr_images = rearrange(color_image, 'h w c -> c h w')
        curr_image = np.stack(curr_images, axis=0)
        curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0).unsqueeze(0)


        ### query policy
        if config['policy_class'] == "ACT":
            if t % query_frequency == 0:
                all_actions = policy(qpos, curr_image)
            if temporal_agg:
                all_time_actions[[t], t:t+num_queries] = all_actions
                actions_for_curr_step = all_time_actions[:, t]
                actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                actions_for_curr_step = actions_for_curr_step[actions_populated]
                k = 0.01
                exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                exp_weights = exp_weights / exp_weights.sum()
                exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
            else:
                raw_action = all_actions[:, t % query_frequency]
        elif config['policy_class'] == "CNNMLP":
            raw_action = policy(qpos, curr_image)
        else:
            raise NotImplementedError

        ### post-process actions
        raw_action = raw_action.squeeze(0).cpu().numpy()
        action = post_process(raw_action)
        target_qpos = action

        ### for visualization
        qpos_list.append(qpos_numpy)
        target_qpos_list.append(target_qpos)
        t+=1



    # print(prediction[5]['world_vector'])
    target_position.x = target_qpos[0]*1000.
    target_position.y = target_qpos[1]*1000.
    target_position.z = target_qpos[2]*1000.
    target_position.rx = target_qpos[3]
    target_position.ry = target_qpos[4]
    target_position.rz = target_qpos[5]
    print('{}:{}'.format('x', target_position.x))
    print('{}:{}'.format('y', target_position.y))
    print('{}:{}'.format('z', target_position.z))
    print('{}:{}'.format('rx', target_position.rx))
    print('{}:{}'.format('ry', target_position.ry))
    print('{}:{}'.format('rz', target_position.rz))
    #object_pub.publish(target_position)
    response=ACTMsgResponse()
    if request.get == True:
        response.x = target_position.x  
        response.y = target_position.y 
        response.z = target_position.z 
        response.gripper = bool(target_qpos[6] > 0)
        response.message = "{}Get obj pos has been executed".format(target_position.x)
    return response


def main():
    rospy.init_node('rt_node',anonymous=True)
    #rt_data_service = rospy.Service('/jaka_driver/camera_image',Image,rt_data_service_callback)  
    object_service = rospy.Service('/act_msg',ACTMsg,act_service_callback)  
    rate = rospy.Rate(3)
    rospy.spin()


if __name__ == '__main__':
    main()

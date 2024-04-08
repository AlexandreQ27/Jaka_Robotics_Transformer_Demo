#!/home/qyb/RT-1/rt-1/bin/python3
import rospy
import cv2
import numpy as np
import pyrealsense2 as rs
import tensorflow as tf
from geometry_msgs.msg import Point
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from jaka_msgs.srv import RTMsg,RTMsgResponse
from robotics_transformer import transformer_network
from tensor2robot.utils import tensorspec_utils
from tf_agents.specs import tensor_spec
import time
import argparse
from robotics_transformer.data_loader import rlds_dataset_loader
import os

def get_args():
    parser = argparse.ArgumentParser(description='获得训练参数')
    parser.add_argument('--vocab_size', '-vs', help='vocab size for discretization', default=256, type=int)  # 离散词典大小
    parser.add_argument('--loaded_checkpoints_dir', '-lcd', help='模型加载目录', default="/home/qyb/RT-1/robotics_transformer/model/", type=str)
    args = parser.parse_known_args()[0]
    return args

def create_model(args):
    '''创建模型'''
    data_target_width = 320
    data_target_height = 256
    
    action_spec = tensorspec_utils.TensorSpecStruct()
    action_spec.world_vector = tensor_spec.BoundedTensorSpec(
        (3,), dtype=tf.float32, minimum=-1., maximum=1., name='world_vector')

    action_spec.rotation_delta = tensor_spec.BoundedTensorSpec(
        (3,),
        dtype=tf.float32,
        minimum=-np.pi / 2,
        maximum=np.pi / 2,
        name='rotation_delta')

    action_spec.gripper_closedness_action = tensor_spec.BoundedTensorSpec(
        (1,),
        dtype=tf.float32,
        minimum=-1.,
        maximum=1.,
        name='gripper_closedness_action')
    action_spec.terminate_episode = tensor_spec.BoundedTensorSpec(
        (2,), dtype=tf.int32, minimum=0, maximum=1, name='terminate_episode')

    state_spec = tensorspec_utils.TensorSpecStruct()
    state_spec.image = tensor_spec.BoundedTensorSpec(
        [data_target_height, data_target_width, 3],
        dtype=tf.float32,
        name='image',
        minimum=0.,
        maximum=1.)
    state_spec.natural_language_embedding = tensor_spec.TensorSpec(
        shape=[512],
        dtype=tf.float32,
        name='natural_language_embedding')
    

    network = transformer_network.TransformerNetwork(
        input_tensor_spec=state_spec,
        output_tensor_spec=action_spec,
        vocab_size=int(args.vocab_size),
        token_embedding_size=512,
        num_layers=8,
        layer_size=128,
        num_heads=8,
        feed_forward_size=512,
        dropout_rate=0.1,
        time_sequence_length=6,
        crop_size=236,
        use_token_learner=True,
        action_order=['terminate_episode', 'world_vector', 'rotation_delta', 'gripper_closedness_action'])
    return network

def load_model(args):
    # 创建一个新模型实例，其结构应与保存的模型一致
    network = create_model(args)

    # 定义检查点目录
    checkpoint_prefix = os.path.join(args.loaded_checkpoints_dir)

    # 找到最新的checkpoint编号
    latest_ckpt_number = max([int(file.split("-")[1].split(".")[0]) 
                                for file in os.listdir(checkpoint_prefix) 
                                if file.startswith("ckpt-")])

    # 构建完整的checkpoint路径
    # full_ckpt_path = os.path.join(checkpoint_prefix, f"ckpt-{latest_ckpt_number}")
    full_ckpt_path = os.path.join(checkpoint_prefix, f"ckpt-39")
    # 检查路径是否存在
    if os.path.exists(full_ckpt_path + ".index"):
        ckpt = tf.train.Checkpoint(model=network)
        
        # 加载模型权重
        status = ckpt.restore(full_ckpt_path)
        status.expect_partial().assert_existing_objects_matched()
        
        print(f"模型从 {full_ckpt_path} 成功加载！")
        return network
    else:
        print("未找到可用的checkpoint文件！")
        return None
    
    return network

def resize(image):
  image = tf.image.resize_with_pad(image, target_width=320, target_height=256)
  image = tf.cast(image, tf.float32)/255.
  return image

def check_array_condition(arr):
    # 检查第一个元素是否为1
    return bool(arr[0])


# 获取训练脚本中的参数
args = get_args()  # 假设你已经实现了get_args函数

# 加载模型
loaded_network = load_model(args)


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

# 获取自然语言嵌入
embed = tf.keras.models.load_model("/home/qyb/rt-x/universal-sentence-encoder_4/")
language_instruction=['Pick up the red square from the table.']
language_embedding = embed(language_instruction).numpy()
natural_language_embedding = language_embedding.reshape(-1)
network_state = tensor_spec.sample_spec_nest(loaded_network.state_spec, outer_dims=[1])

#rt-1 model init


def rt_service_callback(request):
    frames = pipeline.wait_for_frames()

    # 获取彩色图像帧
    color_frame = frames.get_color_frame()
    if not color_frame:
        return
    # 预处理图像
    color_image = np.asanyarray(color_frame.get_data())
    image = tf.convert_to_tensor(color_image, dtype=tf.float32) 
    image = resize(image)

    
    model_input = {
        'image': np.expand_dims(image, axis=0),
        'natural_language_embedding': np.expand_dims(natural_language_embedding, axis=0)
        # 'natural_language_embedding': natural_language_embedding
    }
    print(model_input['image'].shape)
    print(model_input['natural_language_embedding'].shape)

    # 对模型进行推理
    with tf.GradientTape() as tape:  # 在实际推理时不需要GradientTape，这里是为了演示如何计算损失等
        # model(observation_batch, step_type=None, network_state=network_state, training=True)
        prediction = loaded_network(model_input,step_type=None ,network_state=network_state,training=False)
        
    print(prediction)
    # print(prediction[5]['world_vector'])
    target_position.x = prediction[0]['world_vector'][0][0]*1000.
    target_position.y = prediction[0]['world_vector'][0][1]*1000.
    target_position.z = prediction[0]['world_vector'][0][2]*1000.
    print('{}:{}'.format('x', target_position.x))
    print('{}:{}'.format('y', target_position.y))
    print('{}:{}'.format('z', target_position.z))
    #object_pub.publish(target_position)
    response=RTMsgResponse()
    if request.get == True:
        response.x = target_position.x  
        response.y = target_position.y 
        response.z = target_position.z 
        response.gripper = bool(prediction[0]['gripper_closedness_action'][0] > 0)
        response.termination = check_array_condition(prediction[0]['terminate_episode'][0])
        response.message = "{}Get obj pos has been executed".format(target_position.x)
    return response


def main():
    rospy.init_node('rt_node',anonymous=True)
    #rt_data_service = rospy.Service('/jaka_driver/camera_image',Image,rt_data_service_callback)  
    object_service = rospy.Service('/rt_msg',RTMsg,rt_service_callback)  
    rate = rospy.Rate(3)
    rospy.spin()


if __name__ == '__main__':
    main()

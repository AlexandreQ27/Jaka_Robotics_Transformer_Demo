"""
robotic transformer(https://github.com/google-research/robotics_transformer)的多节点分布式训练代码,
采用tensorflow2的distribute.MultiWorkerMirroredStrategy(https://www.tensorflow.org/api_docs/python/tf/distribute/MultiWorkerMirroredStrategy)进行分布式训练，使用加载rlds(https://github.com/google-research/rlds)数据的方式进行数据的读取
使用方法：
    python distribute_worker_train.py --args = param, 其中args见代码中的get_args()
"""

import os
import time
from robotics_transformer.data_loader import rlds_dataset_loader
import tensorflow as tf
import jax
import argparse
import json

'''
Description:
    设置分布式训练参数
Parameters:
    None
Return:
    args:包含训练需要的各种参数，参数详情请见代码
'''
def get_args():
    parser = argparse.ArgumentParser(description='获得分布式训练参数')
    parser.add_argument('--single_gpu_batch_size', '-s', help='batch size for single gpu', default=8, type=int)
    parser.add_argument('--training_epoch', '-te', help='training epoch', default=100, type=int)  # 训练epoch
    parser.add_argument('--log_step', '-ls', help='log step', default=10, type=int)
    parser.add_argument('--dataset_dirs', '-d', help='dataset path', default="/home/qyb/rt/robotic-transformer-pytorch/data")
    parser.add_argument('--learning_rate', '-lr', help='learning rate', default=0.00001, type=float)  # 学习率
    parser.add_argument('--vocab_size', '-vs', help='vocab size for discretization', default=256, type=int)  # 离散词典大小
    parser.add_argument('--dataset_episode_num', '-den', help='训练数据量', default=100, type=int)
    parser.add_argument('--loaded_checkpoints_dir', '-lcd', help='模型加载目录', default="~/", type=str)
    parser.add_argument('--save_model', '-sm', help='save model', default=True)
    parser.add_argument('--model_save_epoch', '-mse', help='save model at every num epoch', default=10, type=int)
    parser.add_argument('--checkpoints_saved_dir', '-csd', help='模型保存目录', default="/home/qyb/rt/robotic-transformer-pytorch/data/model", type=str)
    args = parser.parse_args()
    return args


time_sequence_length = 6  # 常量，来自论文每次预测使用6张图片


def create_train_dataset(args, global_batch_size):
    '''创建数据集'''
    dataset_dirs = args.dataset_dirs.split("+") # 可以包含多个数据集，输入时用+分割

    workdir = "~/"
    sequence_length = time_sequence_length # 每次图片张数
    data_target_width = 456 # 输入图像的宽度
    data_target_height = 256 # 输入图像的高度
    random_crop_factor = 0.95
    replay_capacity = 5_000
    seed = 42
    rng = jax.random.PRNGKey(seed) #jax可以理解为针对硬件的numpy等的加速包，详见项目pdf
    rng, data_rng = jax.random.split(rng)
    data_rng = jax.random.fold_in(data_rng, jax.process_index())

    '''导入rlds类型的数据集，rlds数据集为google论文中给定的数据格式，详见项目pdf'''
    train_ds = rlds_dataset_loader.create_datasets(
        data_rng,
        dataset_dirs=dataset_dirs,
        sequence_length=sequence_length,
        global_batch_size=global_batch_size,
        target_width=data_target_width,
        target_height=data_target_height,
        random_crop_factor=random_crop_factor,
        cache=False,
        shuffle=True,
        shuffle_buffer_size=replay_capacity,
        cache_dir=workdir,
        dataset_episode_num=args.dataset_episode_num
    )

    return train_ds

if __name__ == '__main__':
    os.environ.pop('TF_CONFIG', None) # 清除TF_CONFIG
    args = get_args()
    mirrored_strategy = tf.distribute.MirroredStrategy()
    global_batch_size = args.single_gpu_batch_size *              mirrored_strategy.num_replicas_in_sync
    train_ds = create_train_dataset(args, global_batch_size)
    print(train_ds.__dict__)
    print("dataset")


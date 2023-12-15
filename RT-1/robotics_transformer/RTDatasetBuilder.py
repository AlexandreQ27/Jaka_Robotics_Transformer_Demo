from typing import Iterator, Tuple, Any
import glob
import numpy as np
import tensorflow as tf
import tensorflow_datasets.public_api as tfds
import os
from PIL import Image
# import tensorflow_hub as hub

class RTDatasetBuilder(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._embed = tf.keras.models.load_model("/home/qyb/rt-x/universal-sentence-encoder_4/")

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'image': tfds.features.Image(
                            shape=(480, 640, 3),
                            dtype=np.uint8,
                            doc='RGB observation.',
                        ),
                        'natural_language_embedding': tfds.features.Tensor(
                            shape=(512,),
                            dtype=np.float32,
                            doc='universial sentence embedding instruction',
                        ),
                        "natural_language_instruction": tfds.features.Tensor(
                            shape=(),
                            dtype=np.str_
                        ),
                    }),
                    'action': tfds.features.FeaturesDict({
                       'world_vector': tfds.features.Tensor(
                            shape=(3,),
                            dtype=np.float32,
                            doc='commanded end-effector displacement, in base-relative frame',
                        ),
                        'rotation_delta': tfds.features.Tensor(
                            shape=(3,),
                            dtype=np.float32,
                            doc='rpy commanded orientation displacement, in base-relative frame',
                        ),
                        'terminate_episode': tfds.features.Tensor(
                            shape=(3,),
                            dtype=np.int32,
                        ),
                        'gripper_closedness_action': tfds.features.Tensor(
                            shape=(1,),
                            dtype=np.float32,
                            doc='continuous gripper position'
                        ),
                    }),
                    'reward': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Reward if provided, 1 on final step for demos.'
                    ),
                    'is_first': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on first step of the episode.'
                    ),
                    'is_last': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode.'
                    ),
                    'is_terminal': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode if it is a terminal step, True for demos.'
                    ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.'
                    ),
                }),
            }))
    
    def _split_generators(self,dl_manager):
        """Define data splits."""
        # train=self._generate_examples(path='/home/qyb/RT-1/grasp_data/train/episode_*')
        # val=self._generate_examples(path='/home/qyb/RT-1/grasp_data/val/episode_*')
        # test=self._generate_examples(path='/home/qyb/RT-1/grasp_data/test/episode_*')
        return {
            'train': self._generate_examples(path='/home/qyb/RT-1/grasp_data/train/episode_*'),
            'val': self._generate_examples(path='/home/qyb/RT-1/grasp_data/val/episode_*'),
	        'test': self._generate_examples(path='/home/qyb/RT-1/grasp_data/test/episode_*')
        }
    
    def _generate_examples(self, path) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""
        print("_generate_examples")
        def _parse_example(episode_path):
            # load raw data --> this should change for your dataset
            image_folder_path=episode_path + '/images/'
            txt_folder_path = episode_path

            #def decode_inst(inst):
                #return bytes(inst[np.where(inst != 0)].tolist()).decode("utf-8")
            
            def get_number(file_name):
                return int(file_name.split('.')[0])
            
            # assemble episode --> here we're assuming demos so we set reward to 1 at the end
            episode = []
            with open(txt_folder_path + '/pose.txt', 'r') as file:
                lines = file.readlines()
            # 打开图片文件夹，列出所有图片文件
            images = os.listdir(image_folder_path)
            images.sort(key=get_number)
            assert len(lines)==len(images)
            
            for i,image in enumerate(images):
                if i==0:
                    is_first=np.bool_(True)
                else:
                    is_first=np.bool_(False)
                if i==len(images)-1:
                    is_last=np.bool_(True)
                    is_terminal=np.bool_(True)
                    #reward=np.array([1.], dtype=np.float32)
                    terminate_episode = np.array([1, 0, 0], dtype=np.int32)
                    reward=np.bool_(True)
                else:
                    is_last=np.bool_(False)
                    is_terminal=np.bool_(False)
                    #reward=np.array([0.], dtype=np.float32)
                    terminate_episode = np.array([0, 1, 0], dtype=np.int32)
                    reward=np.bool_(False)
                #language_embedding = self._embed([decode_inst(np.array(step['instruction']))])[0].numpy()
                img = Image.open(os.path.join(image_folder_path, image))
                language_instruction=['Pick up the blue square from the table.']
                language_embedding = self._embed(language_instruction).numpy()
                language_embedding=language_embedding.reshape(-1)
                img_array = np.array(img)
                #print('world_vector',np.float32(lines[i].split())[:3]/1000.)
                episode.append({
                    'observation': {
                        'image': img_array,
                        'natural_language_embedding':language_embedding,
                        'natural_language_instruction': np.str_(language_instruction),
                    },
                    'action': {
                        'world_vector': np.float32(lines[i].split())[:3]/1000.,
                        'terminate_episode': terminate_episode,
                        'rotation_delta': np.float32(lines[i].split())[3:]/180.0*np.pi,
                        # need to save !!!!!!!!!!!!!!!!!!!!!!!!!!!!
                        'gripper_closedness_action': np.array([1.], dtype=np.float32)
                    },
                    'reward': reward,
                    'is_first': is_first,
                    'is_last': is_last,
                    'is_terminal': is_terminal,
                })

            # create output data sample
            sample = {
                'steps': episode,
                'episode_metadata': {
                    'file_path': episode_path
                }
            }

            # if you want to skip an example for whatever reason, simply return None
            return episode_path, sample

        # create list of all examples
        episode_paths = glob.glob(path)

        # for smallish datasets, use single-thread parsing
        for sample in episode_paths:
            yield _parse_example(sample)

        # for large datasets use beam to parallelize data parsing (this will have initialization overhead)
        # beam = tfds.core.lazy_imports.apache_beam
        # return (
        #         beam.Create(episode_paths)
        #         | beam.Map(_parse_example)
        # )


    

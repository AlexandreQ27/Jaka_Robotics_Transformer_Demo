import tensorflow as tf
import jax
import argparse
import json
import tensorflow_datasets as tfds


if __name__ == '__main__':
    
    datasets = tfds.load('RTDatasetBuilder', 
                     split='train', 
                     data_dir='/home/qyb/RT-1/data', 
                     download=True)

    
    # download_and_prepare = dataset_builder.as_dataset(split='train').repeat()
    print(datasets)

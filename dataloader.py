import cv2
import numpy as np
import tensorflow as tf
import os

IMAGE_DIR = '../data/TrainVal_images/Images/'
HUMAN_DIR = '../Human_id/'
CATEGORY_DIR = '../Category_id'
INSTANCE_DIR = '../Instance_id'
TRAIN_ID_LIST = '../data/train_id_demo.txt'
# TRAIN_ID_LIST = ['./data/train_id.txt']
DATA_LIST = '../data/'

def dataload():
    print('This is the data load function of the network')
    filename_queue = tf.train.string_input_producer(TRAIN_ID_LIST,shuffle=False)
    reader = tf.TextLineReader()
    key, value = reader.read(filename_queue)
    print(value)
    example = tf.decode_raw(value)
    print(example)

def read_labeled_image_list(data_dir,human_dir, category_dir, instance_dir, data_id_list):

    image_list = []
    human_list = []
    category_list = []
    instance_list = []

    with open(data_id_list,'r') as f:
        lines = f.readline()
        for line in lines:
            line = line.strip('\n')
            line = line + '.png'
            image_list.append(data_dir + line)
            human_list.append(human_dir + line)
            category_list.append(category_dir + line)
            instance_list.append(instance_dir + line)

    return image_list, human_list, category_list, instance_list


class DataReader(object):

    def __init__(self, data_dir,human_dir, category_dir, instance_dir,data_id_list, input_size, random_scale,
                 random_mirror, shuffle, coord):
        self.data_dir = data_dir
        self.human_dir = human_dir
        self.category_dir = category_dir
        self.instance_dir = instance_dir
        # self.data_list = data_list
        self.data_id_list = data_id_list
        self.input_size = input_size
        self.coord = coord

        self.image_list, self.category_list, self.human_list, self.instance_list = read_labeled_image_list(self.data_dir, self.human_dir, self.category_dir, self.instance_dir self.data_id_list)
        self.images = tf.convert_to_tensor(self.image_list, dtype=tf.string)
        self.human_ids = tf.convert_to_tensor(self.human_list,dtype=tf.string)
        self.categories = tf.convert_to_tensor(self.human_list, dtype=tf.string)
        self.instances = tf.convert_to_tensor(self.instance_list, dtype=tf.string)
        self.queue = tf.train.slice_input_producer([self.images, self.human_ids, self.categories, self.instances], shuffle=False)
        self.image, self.human, self.category, self.instance = read_images_from_disk(self.queue, self.input_size,random_scale, random_mirror)



        print(' Init the data reader')



if __name__=='__main__':

    dataload()
import os
import random
import tensorflow as tf
import cv2 as cv
import numpy as np

data_dir = "G:/1Datasets/catsvsdogs"

train_cats_dir = data_dir + '/train/cats/'
train_dogs_dir = data_dir + '/train/dogs/'

valid_cats_dir = data_dir + '/valid/cats/'
valid_dogs_dir = data_dir + '/valid/dogs/'

train_file_label = "train_img_label.txt"
train_file_label_shuffle = "train_img_label_shuffle.txt"
tfrecord_train_file = "train.tfrecords"

valid_file_label = "val_img_label.txt"
tfrecord_val_file = "val.tfrecords"

# The following functions can be used to convert a value to a type compatible
# with tf.Example.


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# 生成标签文件 -- cat:0  dog:1
def generate_label_file():
    with open(train_file_label, "w") as file1, open(valid_file_label, "w") as file2:
        for filename in os.listdir(train_cats_dir):
            file1.write(train_cats_dir + filename + " " + "0" + "\n")
        for filename in os.listdir(train_dogs_dir ):
            file1.write(train_dogs_dir  + filename + " " + "1" + "\n")

        for filename in os.listdir(valid_cats_dir):
            file2.write(valid_cats_dir + filename + " " + "0" + "\n")
        for filename in os.listdir(valid_dogs_dir):
            file2.write(valid_dogs_dir + filename + " " + "1" + "\n")
    file1.close()
    file2.close()


# 事先打乱标签文件,训练用标签文件
def shuflle_file_line():
    with open(train_file_label, "r") as file1:
        lines = file1.readlines()
    file1.close()
    random.shuffle(lines)
    with open(train_file_label_shuffle, "w") as file2:
        for line in lines:
            file2.write(line)
    file1.close()


# 从标签文件中获取图像名及标签
def get_imgs_labels(is_train_file = True):
    filenames = []
    labels = []
    if is_train_file:
        with open(train_file_label_shuffle, "r") as f:
            lines = f.readlines()
            for line in lines:
                filenames.append(line.split()[0])
                labels.append(line.split()[1])
        f.close()
    else:
        with open(valid_file_label, "r") as f:
            lines = f.readlines()
            for line in lines:
                filenames.append(line.split()[0])
                labels.append(line.split()[1])
        f.close()
    return filenames, labels


# 生成tfrecord文件
def gennerate_terecord_file(is_train_file = True):
    cnt = 0
    filenames, labels = get_imgs_labels(is_train_file)
    if is_train_file:
        write_tfrecord_flie = tfrecord_train_file
    else:
        write_tfrecord_flie = tfrecord_val_file
    with tf.io.TFRecordWriter(write_tfrecord_flie) as writer:
        for filename, label in zip(filenames, labels):
            cnt = cnt + 1
            image_bgr = cv.imread(filename)
            image_resized = cv.resize(image_bgr, (112, 112))
            image_gray = cv.cvtColor(image_resized, cv.COLOR_BGR2GRAY)
            image = image_gray.tostring()
            label = int(label)
            feature = {  # 建立 tf.train.Feature 字典
                'image': _bytes_feature(image),  # 图片是一个 Bytes 对象
                'label': _int64_feature(label) # 标签是一个 Int 对象
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))  # 通过字典建立 Example
            writer.write(example.SerializeToString())  # 将Example序列化并写入 TFRecord 文件
            if cnt % 100 == 0:
                print("the length of tfrecord is %d" %cnt)


# 将 TFRecord 文件中的每一个序列化的 tf.train.Example 解码
def _parse_example(example_string):
    # 定义Feature结构，告诉解码器每个Feature的类型是什么
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
    }
    feature_dict = tf.io.parse_single_example(example_string, feature_description)

    images = feature_dict['image']
    images = tf.io.decode_raw(images, tf.uint8)
    images = tf.reshape(images, [112, 112, 1])  # 灰度图
    images = tf.cast(images, tf.float32) / 255.0

    labels = feature_dict['label']
    labels = tf.cast(labels, tf.int64)

    return images, labels


# 读取训练用TFRecord文件,一批数据
def gen_train_data_batch(file_pattern, batch_size, num_repeat):
    files = tf.data.Dataset.list_files(file_pattern)
    dataset = files.flat_map(tf.data.TFRecordDataset)
    dataset = dataset.repeat(num_repeat)
    dataset = dataset.map(_parse_example, num_parallel_calls=4)
    dataset = dataset.batch(batch_size)
    dataset = dataset.shuffle(buffer_size=16 * batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset


def gen_val_data_batch(file_pattern, batch_size):
    files = tf.data.Dataset.list_files(file_pattern)
    dataset = files.flat_map(tf.data.TFRecordDataset)
    dataset = dataset.map(_parse_example, num_parallel_calls=4)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset


if __name__ == "__main__":
    """依次运行以下函数"""
    # generate_label_file()
    # shuflle_file_line()
    # gennerate_terecord_file(is_train_file = False)
    dataset1 = gen_val_data_batch(file_pattern="val.tfrecords", batch_size=64)
    for batch, (images1, labels1) in enumerate(dataset1):
        # print(images.get_shape())
        for k in range(10):
            img = tf.image.encode_png(images1[k])
            lable = labels1[k].numpy()
            pngname = "./val_img_test/" + str(k) + "_" + str(lable) + ".png"
            with tf.io.gfile.GFile(pngname, 'wb') as file:
                file.write(img.numpy())

import glob
import numpy as np
np.random.seed(1)
import sys
import tensorflow as tf
from matplotlib.image import imread
import matplotlib.pyplot as plt
from pandas.io.parsers import read_csv
from random import shuffle
import cv2
import pandas as pd
from PIL import Image

dir = "d:/MTSD/copy/"
image_dir = dir + 'Detection/'
train_filename = 'train.tfrecords'  # address to save the TFRecords file
val_filename = 'val.tfrecords'  # address to save the TFRecords file
test_filename = 'test.tfrecords'  # address to save the TFRecords file


def read_file():
    return read_csv(dir + "GT.csv", sep=',', skiprows=1, header=None, names=["filename", "Class ID" , "xmin", "ymin", "xmax",
                                                                      "ymax"])


def read_translation():
    return read_csv(dir + "class_translation.txt", sep=';', header=None, names=["index", "class_name"])


def class_distribution(labels):
    sign_classes, class_indices, class_counts = np.unique(labels, return_index=True, return_counts=True)
    num_classes = sign_classes.shape[0]
    x_tix = range(sign_classes[0], num_classes + 2)
    bins = np.arange(num_classes + 2) - 0.5
    fig = plt.figure(figsize=(25,8))
    plt.style.use('fivethirtyeight')
    ax1 = fig.add_subplot(1,1,1)
    plt.hist(labels, bins)
    plt.title("Number of samples per different sign type", loc='center')
    plt.xlabel("Traffic Sign class/ids")
    plt.ylabel("Samples ")
    plt.xticks(np.arange(min(x_tix), max(x_tix), 1.0), rotation=90, fontsize=10)
    plt.show()


def class_distribution_ordered(labels):
    sign_classes, class_indices, class_counts = np.unique(labels, return_index=True, return_counts=True)
    num_classes = sign_classes.shape[0]
    class_counts, sign_classes = zip(*sorted(zip(class_counts, sign_classes)))
    fig = plt.figure(figsize=(25,8))
    plt.style.use('fivethirtyeight')
    ax1 = fig.add_subplot(1,1,1)
    x = range(0, num_classes)
    plt.xticks(x, sign_classes, rotation=90, fontsize=10)
    plt.bar(x, class_counts)
    plt.title("Number of samples per different sign type", loc='center')
    plt.xlabel("Traffic Sign class/ids")
    plt.ylabel("Samples ")
    plt.show()


def parse_img_addr(addrs):
    return [image_dir + addr.replace("'", "") for addr in addrs]


def train_test(signs, shuffle_data):
    addrs = signs['File Name'].get_values()
    addrs = parse_img_addr(addrs)
    labels = signs['Class ID'].get_values()
    # to shuffle data
    if shuffle_data:
        c = list(zip(addrs, labels))
        shuffle(c)
        addrs, labels = zip(*c)

    # Divide the hata into 80% train, 20% and 20% test
    train_addrs = addrs[0:int(0.8 * len(addrs))]
    train_labels = labels[0:int(0.8 * len(labels))]
    test_addrs = addrs[int(0.8 * len(addrs)):]
    test_labels = labels[int(0.8 * len(labels)):]
    return train_addrs, train_labels, test_addrs, test_labels


def draw_boxes(image_name):
    full_labels = read_file()
    full_labels.head()
    selected_value = full_labels[full_labels.filename == image_name]
    img = cv2.imread(image_dir + '{}'.format(image_name))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for index, row in selected_value.iterrows():
        img = cv2.rectangle(img, (row['xmin'], row['ymin']), (row['xmax'], row['ymax']), (0, 255, 0), 1)
    return img


def filter_file_by_class_count(file, filter=20):
    sign_classes, class_indices, class_counts = np.unique(file['Class ID'].get_values(), return_index=True, return_counts=True)
    sign_classes, class_counts = zip(*((class_id, count) for class_id, count in zip(sign_classes, class_counts) if count >= filter))
    return file[file['Class ID'].isin(sign_classes)]


def create_count_file(file, translation):
    sign_classes, class_indices, class_counts = np.unique(file['Class ID'].get_values(), return_index=True,
                                                          return_counts=True)
    class_counts, sign_classes, sign_names = zip(*sorted(zip(class_counts, sign_classes, translation['class_name'].get_values()), reverse=True))
    np.savetxt("class_counts.txt", np.column_stack((sign_classes, sign_names, class_counts)), delimiter=";", fmt='%s')

if __name__ == '__main__':
    full_labels = read_file()
    full_labels.head()
    Image.fromarray(draw_boxes(full_labels['filename'].get_values()[2])).show()


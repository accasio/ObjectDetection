import glob
import numpy as np
np.random.seed(1)
import sys
import os
import tensorflow as tf
from matplotlib.image import imread
import matplotlib.pyplot as plt
import random
random.seed(0)
from random import shuffle
from pandas.io.parsers import read_csv
import cv2
import pandas as pd
from PIL import Image

dir = "c:/MTSD/Updated/"
image_dir = dir + 'Detection/'
train_filename = 'train.tfrecords'  # address to save the TFRecords file
val_filename = 'val.tfrecords'  # address to save the TFRecords file
test_filename = 'test.tfrecords'  # address to save the TFRecords file


def read_file():
    return read_csv(dir + "train.csv", sep=',', header=None, names=["filename", "xmin", "ymin", "xmax",
                                                                      "ymax", "Class ID"])


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


def train_test_ordered(labels, train, test):
    sign_classes, _, class_counts = np.unique(labels, return_index=True, return_counts=True)
    train_classes, _, train_counts = np.unique(train, return_index=True, return_counts=True)
    test_classes, _, test_counts = np.unique(train, return_index=True, return_counts=True)

    num_classes = sign_classes.shape[0]
    class_counts, sign_classes = zip(*sorted(zip(class_counts, sign_classes)))
    class_counts, train_counts = zip(*sorted(zip(class_counts, train_counts)))
    class_counts, test_counts = zip(*sorted(zip(class_counts, test_counts)))
    fig = plt.figure(figsize=(25,8))
    plt.style.use('fivethirtyeight')
    ax1 = fig.add_subplot(1,1,1)
    x = range(0, num_classes)
    plt.xticks(x, sign_classes, rotation=90, fontsize=10)
    p1 = plt.bar(x, train_counts, color='#d62728')
    p2 = plt.bar(x, class_counts, color='blue')
    # plt.bar(x, test_counts)
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
    sign_names = []
    for s_class in sign_classes:
        t = translation.loc[translation['index'] == s_class]['class_name'].item()
        sign_names.append(t)

    class_counts, sign_classes, sign_names = zip(*sorted(zip(class_counts, sign_classes, sign_names), reverse=True))
    np.savetxt("class_counts.txt", np.column_stack((sign_classes, sign_names, class_counts)), delimiter=";", fmt='%s')

def data_gen(file):
    image1 = file.iloc[0]
    # img = np.asarray(Image.open('C:/mtsd/detection/' + image1['filename']).convert('L'))
    image_contents = tf.read_file('C:/mtsd/updated/detection/' + image1['filename'])
    img = tf.image.decode_image(image_contents, channels=3)
    img = tf.cast(img, tf.float32)

    bounding_boxes = [image1['ymin'], image1['xmin'], image1['ymax'], image1['xmax']]

    bounding_boxes = tf.constant(bounding_boxes,
                       dtype=tf.float32,
                       shape=[1, 1, 4])

    bbox = tf.constant([1, 1, 4],
                       dtype=tf.float32,
                       shape=[1, 1, 3])

    begin, size, bbox_for_draw = tf.image.sample_distorted_bounding_box(
        tf.shape(img),
        use_image_if_no_bounding_boxes=True,
        bounding_boxes=None)

    # Draw the bounding box in an image summary.
    image_with_box = tf.image.draw_bounding_boxes(tf.expand_dims(img, 0),
                                                  bbox_for_draw)
    tf.summary.image('images_with_box', image_with_box)

    # Employ the bounding box to distort the image.
    distorted_image = tf.slice(img, begin, size)

    # sess = tf.Session()
    # with sess.as_default():
    #     print(type(distorted_image))
    init_op = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init_op)  # execute init_op
        # print(type(np.asarray(distorted_image)))
        # distorted_image = tf.reshape(distorted_image, image1.shape)
        array = distorted_image.eval(session=sess)

        print(array)
        # Image.fromarray(np.ndarray(distorted_image).astype('uint8')).show()


def move_col_to_back(df, col_to_move):
    cols = list(df.columns.values)  # Make a list of all of the columns in the df
    temp = cols[col_to_move]
    cols.pop(col_to_move)
    cols.append(temp)
    # temp = cols[col_move_to]
    # cols[col_move_to] = cols[col_to_move]
    # cols[col_to_move] = temp
    return df[cols]


def convert_ids_to_names(file, translation):
    cols = list(file.columns.values)
    index = cols.index('Class ID')
    names = []
    for idx, id in enumerate(file['Class ID'].get_values()):
        new = translation.loc[translation['id'] == id]
        names.append(new['class_name'].iat[0])

    n = file.columns[index]
    file.drop(n, axis=1, inplace=True)
    file[n] = names

    return file

def create_train_test(df, ratio=0.8):
    uniq = df['filename'].unique()
    random.shuffle(uniq)
    split = int(len(uniq) * ratio)
    train_names = uniq[:split]
    test_names = uniq[split:]
    train = pd.DataFrame(columns=["filename", "Class ID", "xmin", "ymin", "xmax", "ymax"])
    test = pd.DataFrame(columns=["filename", "Class ID", "xmin", "ymin", "xmax", "ymax"])
    for filename in test_names:
        val = df.loc[df['filename'] == filename]
        test = test.append(val)
    for filename in train_names:
        val = df.loc[df['filename'] == filename]
        train = train.append(val)
    return train, test


def video_to_frames():
    fps = 30
    times = [[94, 97], [174, 195]]
    cap = cv2.VideoCapture('G:/Downloads/Driving with g1w in Malaysia/Driving with g1w in Malaysia (720p_30fps_H264-192kbit_AAC).mp4')

    for time in times:
        directory = str(time[0])
        if not os.path.exists(directory):
            os.makedirs(directory)
        cap.set(cv2.CAP_PROP_POS_MSEC, time[0] * 1000)  # just cue to 20 sec. position
        success = True
        count = 0
        limit = (time[1] - time[0]) * fps
        while success:
            if count <= limit:
                success, frame = cap.read()
                if success:
                    cv2.imwrite(os.path.join(directory, '%d.png') % count, frame)  # save frame as JPEG file
            else:
                success = False
            count = count + 1




if __name__ == '__main__':
    # full_labels = read_file()
    # full_labels.head()
    # Image.fromarray(draw_boxes(full_labels['filename'].get_values()[2])).show()
    video_to_frames()

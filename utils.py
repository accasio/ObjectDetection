import glob
import numpy as np
np.random.seed(1)
import sys
import os
import itertools
import tensorflow as tf
from matplotlib.image import imread
import matplotlib.pyplot as plt
import random
random.seed(0)
from random import shuffle
from pandas.io.parsers import read_csv
from sklearn.metrics import confusion_matrix
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


def images_to_video():
    image_folder = './94/'
    video_name = 'video.avi'

    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, -1, 1, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    code in part taken from http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize == True and cm[i, j] == 0:
            fmt = '.0f'
        elif normalize == True and cm[i, j] != 0:
            fmt = '.2f'
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def conf_matrix():
    y_true = ['Traffic lights ahead', 'Crossroads', 'Obstacles ahead', 'split way', 'Obstacles ahead', 'split way', 'Obstacles ahead', 'split way', 'Stop', 'Hump ahead', 'Speed limit 90', 'split way', 'Hump ahead', 'Minor road on left', 'H. lim. sign 5.-m', 'School children crossing 1', 'Obstacles ahead', 'split way', 'Caution! Hump', 'School children crossing 1', 'Crossroads to the right', 'Caution! Hump', 'split way', 'Obstacles ahead', 'split way', 'No Stopping', 'Crossroads to the right', 'Minor road on left', 'School children crossing 1', 'Traffic lights ahead', 'H. lim. sign 4.-m', 'No Stopping', 'Speed limit 80', 'Obstacles ahead', 'split way', 'Obstacles ahead', 'split way', 'No entry', 'No entry', 'Stop', 'split way', 'Obstacles ahead', 'split way', 'Obstacles ahead', 'split way', 'Speed limit 30', 'Towing zone', 'Pedestrian crossing 2', 'split way', 'Obstacles ahead', 'split way', 'Minor road on left', 'Speed limit 50', 'Minor road on left', 'Speed limit 90', 'Speed limit 90', 'split way', 'Left bend', 'Obstacles ahead', 'split way', 'Towing zone', 'Obstacles ahead', 'split way', 'Obstacles ahead', 'split way', 'H. lim. sign 5.-m', 'Speed limit 30', 'No parking', 'Obstacles ahead', 'split way', 'Speed limit 50', 'School children crossing 1', 'Speed limit 90', 'Speed limit 90', 'Towing zone', 'Minor road on left', 'Obstacles ahead', 'split way', 'Speed limit 50', 'No U-turn', 'Traffic lights ahead', 'H. lim. sign 5.-m', 'Obstacles ahead', 'split way']


    y_pred = ['Pedestrian crossing 1', 'split way', 'Obstacles ahead', 'split way', 'Obstacles ahead', 'split way', 'Obstacles ahead', 'split way', 'U-turn', 'Hump ahead', 'No Stopping', 'Caution! Hump', 'Towing zone', 'Minor road on left', 'Speed limit 60', 'School children crossing 1', 'Obstacles ahead', 'split way', 'Caution! Hump', 'School children crossing 1', 'Crossroads to the right', 'Caution! Hump', 'Pedestrian crossing 2', 'Obstacles ahead', 'split way', 'No Stopping', 'Caution! Hump', 'Minor road on left', 'School children crossing 1', 'Pedestrian crossing 1', 'Speed limit 80', 'No Stopping', 'Speed limit 60', 'Traffic lights ahead', 'split way', 'Obstacles ahead', 'split way', 'Traffic lights ahead', 'No entry', 'Stop', 'Caution! Hump', 'Obstacles ahead', 'split way', 'Obstacles ahead', 'split way', 'Speed limit 60', 'Hump ahead', 'Pedestrian crossing 2', 'split way', 'Obstacles ahead', 'split way', 'split way', 'No Stopping', 'Pedestrian crossing 1', 'Speed limit 80', 'Speed limit 50', 'split way', 'Minor road on left', 'Obstacles ahead', 'Caution! Hump', 'Caution! Hump', 'Obstacles ahead', 'split way', 'Obstacles ahead', 'split way', 'H. lim. sign 5.-m', 'U-turn', 'No parking', 'Obstacles ahead', 'split way', 'Speed limit 80', 'School children crossing 1', 'Speed limit 80', 'Speed limit 80', 'Towing zone', 'Minor road on left', 'Obstacles ahead', 'split way', 'No Stopping', 'Speed limit 60', 'Traffic lights ahead', 'Speed limit 60', 'Obstacles ahead', 'split way']



    plt.rcParams.update({'font.size': 3.5})

    cnf_matrix = confusion_matrix(y_true, y_pred, labels=["Obstacles ahead", "H. lim. sign 5.-m", "Give way", "Minor road on left", "U-turn", "Caution! Hump",
                  "Keep left", "No entry", "Stop", "No Stopping", "Traffic lights ahead", "Speed limit 80",
                  "Keep right", "Crossroads to the left", "Speed limit 60", "Speed limit 50",
                  "School children crossing 1", "Hump ahead", "No parking", "Towing zone", "Caution!",
                  "H. lim. sign 4.-m", "Pedestrian crossing 2", "Road works", "Speed limit 30",
                  "Compulsory motor-cycles track", "Camera operated zone", "Crossroads to the right",
                  "Speed limit 90", "No U-turn", "H. lim. sign 2.-m", "Roundabout ahead", "Narrow roads on the left",
                  "Crossroads", "Pedestrian crossing 1", "Speed limit 110", "H. lim. sign 3.-m", "Left bend",
                  "split way"])
    class_names = "Obstacles ahead", "H. lim. sign 5.-m", "Give way", "Minor road on left", "U-turn", "Caution! Hump", \
                  "Keep left", "No entry", "Stop", "No Stopping", "Traffic lights ahead", "Speed limit 80", \
                  "Keep right", "Crossroads to the left", "Speed limit 60", "Speed limit 50", \
                  "School children crossing 1", "Hump ahead", "No parking", "Towing zone", "Caution!", \
                  "H. lim. sign 4.-m", "Pedestrian crossing 2", "Road works", "Speed limit 30", \
                  "Compulsory motor-cycles track", "Camera operated zone", "Crossroads to the right", \
                  "Speed limit 90", "No U-turn", "H. lim. sign 2.-m", "Roundabout ahead", "Narrow roads on the left", \
                  "Crossroads", "Pedestrian crossing 1", "Speed limit 110", "H. lim. sign 3.-m", "Left bend", \
                  "split way"
    # plt.figure()
    # plot_confusion_matrix(cnf_matrix, classes=class_names,
    #                       title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                          title='Normalized confusion matrix')
    plt.savefig('small.png', format='png', dpi=4000)

if __name__ == '__main__':
    # full_labels = read_file()
    # full_labels.head()
    # Image.fromarray(draw_boxes(full_labels['filename'].get_values()[2])).show()
    conf_matrix()

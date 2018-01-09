from PIL import Image
import numpy as np
import tensorflow as tf



def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={

    })
    image = tf.decode_raw(features['image/encoded'], tf.uint8)
    label = tf.cast(features['image/object/class/label'], tf.int32)
    height = tf.cast(features['image/height'], tf.int32)
    width = tf.cast(features['image/width'], tf.int32)
    return image, label, height, width


def get_all_records(FILE):
    with tf.Session() as sess:
        filename_queue = tf.train.string_input_producer([ FILE ])
        image, label, height, width = read_and_decode(filename_queue)
        bytz = tf.stack([width, height, 3])
        image = tf.reshape(image, bytz)
        image.set_shape([720,720,3])
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for i in range(100):
            example, l = sess.run([image, label])
            img = Image.fromarray(example, 'RGB')
            img.save( "output/" + str(i) + '-train.png')

            print (example,l)
        coord.request_stop()
        coord.join(threads)



def parse(serialized):
    # Define a dict with the data-names and types we expect to
    # find in the TFRecords file.
    # It is a bit awkward that this needs to be specified again,
    # because it could have been written in the header of the
    # TFRecords file instead.
    features = \
        {
            'image/height': tf.FixedLenFeature([], tf.int64),
            'image/width': tf.FixedLenFeature([], tf.int64),
            'image/filename':  tf.FixedLenFeature([], tf.string),
            'image/source_id':  tf.FixedLenFeature([], tf.string),
            'image/encoded':  tf.FixedLenFeature([], tf.string),
            'image/format':  tf.FixedLenFeature([], tf.string),
            'image/object/bbox/xmin':  tf.VarLenFeature(tf.float32),
            'image/object/bbox/xmax': tf.VarLenFeature(tf.float32),
            'image/object/bbox/ymin': tf.VarLenFeature(tf.float32),
            'image/object/bbox/ymax': tf.VarLenFeature(tf.float32),
            'image/object/class/text': tf.VarLenFeature(tf.string),
            'image/object/class/label': tf.VarLenFeature(tf.int64)
        }

    # Parse the serialized data so we get a dict with our data.
    parsed_example = tf.parse_single_example(serialized=serialized,
                                             features=features)

    # Get the image as raw bytes.
    image_raw = parsed_example['image/encoded']

    # Decode the raw bytes so it becomes a tensor with type.
    image = tf.decode_raw(image_raw, tf.uint8)

    # The type is now uint8 but we need it to be float.
    image = tf.cast(image, tf.float32)

    # Get the label associated with the image.
    label = parsed_example['image/object/class/label']
    height = parsed_example['image/height']
    width = parsed_example['image/width']

    # The image and label are now correct TensorFlow types.
    return image, label, height, width


def input_fn(filenames, train, batch_size=32, buffer_size=2048):
    # Args:
    # filenames:   Filenames for the TFRecords files.
    # train:       Boolean whether training (True) or testing (False).
    # batch_size:  Return batches of this size.
    # buffer_size: Read buffers of this size. The random shuffling
    #              is done on the buffer, so it must be big enough.

    # Create a TensorFlow Dataset-object which has functionality
    # for reading and shuffling data from TFRecords files.
    dataset = tf.data.TFRecordDataset(filenames=filenames)

    # Parse the serialized data in the TFRecords files.
    # This returns TensorFlow tensors for the image and labels.
    dataset = dataset.map(parse)

    if train:
        # If training then read a buffer of the given size and
        # randomly shuffle it.
        dataset = dataset.shuffle(buffer_size=buffer_size)

        # Allow infinite reading of the data.
        num_repeat = None
    else:
        # If testing then don't shuffle the data.

        # Only go through the data once.
        num_repeat = 1

    # Repeat the dataset the given number of times.
    dataset = dataset.repeat(num_repeat)

    # Get a batch of data with the given size.
    dataset = dataset.batch(batch_size)

    # Create an iterator for the dataset and the above modifications.
    iterator = dataset.make_one_shot_iterator()

    # Get the next batch of images and labels.
    images_batch, labels_batch = iterator.get_next()

    # The input-function must return a dict wrapping the images.
    x = {'image': images_batch}
    y = labels_batch

    return x, y


def train_input_fn():
    return input_fn(filenames='data/testtrain.record', train=True)


if __name__ == '__main__':
    train_input_fn()
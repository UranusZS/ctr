import os
import sys
import codecs
import logging
import tensorflow as tf 

logger = logging.getLogger("TFRecSYS")
sh = logging.StreamHandler(stream=None)
logger.setLevel(logging.DEBUG)
fmt = "%(asctime)-15s %(levelname)s %(filename)s %(lineno)d %(process)d %(message)s"
datefmt = "%a %d %b %Y %H:%M:%S"
formatter = logging.Formatter(fmt, datefmt)
sh.setFormatter(formatter)
logger.addHandler(sh)

os.environ["CUDA_VISIBLE_DEVICES"] = "2"


def print_info(str_):
    s = "---------- {0} ----------".format(str_)
    print(s)


def tfrecord_input_fn(filenames, shuffle=False, batch_size=32, num_epochs=2):
    dataset = tf.data.TFRecordDataset(filenames)
    def parser(record):
        features_map = {
            "label": tf.FixedLenFeature([], tf.float32),
            "feature_ids": tf.VarLenFeature(tf.int64),
            "feature_vals": tf.VarLenFeature(tf.float32)
        }
        #parsed = tf.parse_example(record, features=features_map)
        parsed = tf.parse_single_example(record, features=features_map)
        label = parsed["label"]
        #ids = parsed["feature_ids"]
        #vals = parsed["feature_vals"]
        return parsed, label
    dataset = dataset.map(parser)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(num_epochs)
    #iterator = dataset.make_initializable_iterator() # Ensure that you have run the initializer operation for this iterator before getting the next element.
    iterator = dataset.make_one_shot_iterator()
    batch_features, batch_labels = iterator.get_next()
    return batch_features, batch_labels
    

if __name__ == "__main__":
    filenames = ["./data/kdda_tfrecord/kdda_00001.tfrecord"]
    next_batch = tfrecord_input_fn(filenames, True)
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    with tf.Session() as sess:
        sess.run(init_op)
        print(sess.run(next_batch))

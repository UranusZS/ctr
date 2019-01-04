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


class LibSVMInputReader(object):
    def __init__(self, file_queue, batch_size, capacity, min_after_dequeue):
        self.file_queue = file_queue
        self.batch_size = batch_size
        self.capacity = capacity
        self.min_after_dequeue = min_after_dequeue
        
    def read(self):
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(self.file_queue)
        shuffle_batch_example = tf.train.shuffle_batch([serialized_example],
                                        batch_size=self.batch_size, capacity=self.capacity,
                                        min_after_dequeue=self.min_after_dequeue)
        features_map = {
            "label": tf.FixedLenFeature([], tf.float32),
            "feature_ids": tf.VarLenFeature(tf.int64),
            "feature_vals": tf.VarLenFeature(tf.float32)
        }
        features = tf.parse_example(shuffle_batch_example, features=features_map)
        batch_label = features["label"]
        batch_feature_ids = features["feature_ids"]
        batch_feature_vals = features["feature_vals"]
        return batch_label, batch_feature_ids, batch_feature_vals 


def reader_test(filenames, batch_size=32, num_epochs=2):
    dataset = tf.data.TFRecordDataset(filenames)
    print_info("dataset {}".format(dataset))
    def parser(record):
        features_map = {
            "label": tf.FixedLenFeature([], tf.float32),
            "feature_ids": tf.VarLenFeature(tf.int64),
            "feature_vals": tf.VarLenFeature(tf.float32)
        }
        #parsed = tf.parse_example(record, features=features_map)
        parsed = tf.parse_single_example(record, features=features_map)
        label = parsed["label"]
        ids = parsed["feature_ids"]
        vals = parsed["feature_vals"]
        return label, ids, vals
    dataset = dataset.map(parser)
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(num_epochs)
    print_info("dataset {}".format(dataset))
    iterator = dataset.make_initializable_iterator()
    print_info("iterator {}".format(iterator))
    
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    next_element = iterator.get_next()
    with tf.Session() as sess:
        print_info("sess start")
        sess.run(init_op)
        sess.run(iterator.initializer)
        print(next_element)
        print_info("next element")
        print(sess.run(next_element))
        print_info("next element")
        #next_element = iterator.get_next()  # no need to call, its an op
        print(sess.run(next_element))
        print_info("sess end")
        
        
if __name__ == "__main__":
    filenames = ["./data/kdda_tfrecord/kdda_00001.tfrecord"]
    batch_size=32
    num_epochs = 2
    reader_test(filenames, batch_size, num_epochs)

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


def reader_test():
    filename_queue = tf.train.string_input_producer(["./data/kdda_tfrecord/kdda_00001.tfrecord"])
    libsvm_input_reader = LibSVMInputReader(filename_queue, 2, 1000, 200)
    init_op = tf.group(tf.initialize_all_variables(), tf.initialize_local_variables())
    batch_label, batch_feature_ids, batch_feature_vals = libsvm_input_reader.read()
    with tf.Session() as sess:
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            step = 0
            while not coord.should_stop():
                step += 1
                print("step: {0}".format(step))
                batch_label_list, batch_feature_ids_list, batch_feature_vals_list = sess.run([batch_label, batch_feature_ids, batch_feature_vals])
                print(batch_label_list)
        except tf.errors.OutofRangeError:
            print("Done")
        finally:
            coord.request_stop()
            coord.join(threads)

if __name__ == "__main__":
    reader_test()

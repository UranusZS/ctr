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

class LibSVM2TFRecord(object):
    def __init__(self, libsvm_filenames, tfrecord_dir, 
                 info_interval=10000, tfrecord_large_line_num=10000000):
        self.libsvm_filenames = libsvm_filenames
        self.tfrecord_dir = tfrecord_dir
        self.info_interval = info_interval
        self.tfrecord_large_line_num = tfrecord_large_line_num
    
    def set_transform_files(self, libsvm_filenames, tfrecord_dir):
        self.libsvm_filenames = libsvm_filename
        self.tfrecord_dir = tfrecord_dir
    
    def _parse_line(self, line, separator=" "):
        label = 0
        feature_ids = []
        vals = []
        error_flag = 0
        line_components = line.strip().split(" ")
        try:
            label = float(line_components[0])
            features = line_components[1:]
        except IndexError:
            error_flag = 1
        except ValueError:
            error_flag = 1
        for feature in features:
            feature_components = feature.split(":")
            try:
                feature_id = int(feature_components[0])
                val = float(feature_components[1])
            except IndexError:
                error_flag = 1
                continue
            except ValueError:
                error_flag = 1
                continue
            feature_ids.append(feature_id)
            vals.append(val)
        return error_flag, [label, feature_ids, vals]
    
    def fit(self):
        logger.info(self.libsvm_filenames)
        if not tf.gfile.Exists(self.tfrecord_dir):
            tf.gfile.MakeDirs(self.tfrecord_dir)
        tfrecord_num = 1
        for libsvm_filename in self.libsvm_filenames:
            logger.info("Begin to process {0}".format(libsvm_filename))
            tfrecord_id = 1
            with codecs.open(libsvm_filename, mode="r", encoding="utf-8") as fread:
                line = fread.readline()
                line_num = 0
                while line:
                    if line_num % self.tfrecord_large_line_num == 0:
                        tfrecord_filename = self.tfrecord_dir + "/" + os.path.basename(libsvm_filename) + "_%05d.tfrecord"%tfrecord_id
                        logger.info(tfrecord_filename)
                        writer = tf.python_io.TFRecordWriter(tfrecord_filename)
                        tfrecord_num += 1
                        tfrecord_id += 1
                        logger.info("Start writing to the tfrecord file {0}".format(tfrecord_filename))
                    line_num += 1
                    error_flag, result = self._parse_line(line, " ")
                    if not error_flag:
                        tfrecord_feature = {
                            "label" : tf.train.Feature(float_list=tf.train.FloatList(value=[result[0]])),
                            "feature_ids": tf.train.Feature(int64_list=tf.train.Int64List(value=result[1])),
                            "feature_vals": tf.train.Feature(float_list=tf.train.FloatList(value=result[2]))
                        }
                        example = tf.train.Example(features=tf.train.Features(feature=tfrecord_feature))
                        writer.write(example.SerializeToString())
                        #logger.info("Info, line_num {0} result: {1}".format(line_num, result))
                    else:
                        logger.info("Error, line_num {0} line: {1}".format(line_num, line))
                    line = fread.readline()
                writer.close()
                logger.info("finished, line_num {0} line: {1}".format(line_num, line))
                logger.info("libsvm: {0} transform to tfrecord: {1} successfully".format(libsvm_filename, tfrecord_filename))

if __name__ == "__main__": 
    libsvm_filenames = ["./data/kdda", "./data/kdda.t"]
    tfrecord_dir = "./data/kdda_tfrecord"
    libsvm_to_tfrecord = LibSVM2TFRecord(libsvm_filenames, tfrecord_dir)
    libsvm_to_tfrecord.fit()

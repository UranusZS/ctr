import os
import sys
import codecs
import logging
import tensorflow as tf 

from reader import *

logger = logging.getLogger("TFRecSYS")
sh = logging.StreamHandler(stream=None)
logger.setLevel(logging.DEBUG)
fmt = "%(asctime)-15s %(levelname)s %(filename)s %(lineno)d %(process)d %(message)s"
datefmt = "%a %d %b %Y %H:%M:%S"
formatter = logging.Formatter(fmt, datefmt)
sh.setFormatter(formatter)
logger.addHandler(sh)

os.environ["CUDA_VISIBLE_DEVICES"] = "2"


def export_model_predict(export_dir, filenames):
    features_map = {
            "label": tf.FixedLenFeature([], tf.float32),
            "feature_ids": tf.VarLenFeature(tf.int64),
            "feature_vals": tf.VarLenFeature(tf.float32)
    }
    def _parse_line(line, separator=" "):
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
    def predict(line, sess):
        error_flag, result = _parse_line(line)
        tfrecord_feature = {
            "label" : tf.train.Feature(float_list=tf.train.FloatList(value=[result[0]])),
            "feature_ids": tf.train.Feature(int64_list=tf.train.Int64List(value=result[1])),
            "feature_vals": tf.train.Feature(float_list=tf.train.FloatList(value=result[2]))
        }
        example = tf.train.Example(features=tf.train.Features(feature=tfrecord_feature))
        # here use the parsing 
        model_input = example.SerializeToString()
        # the tensor names can be get from the saved_model_cli tool
        prediction = sess.run('Sigmoid:0', feed_dict={"input_example_tensor:0": [model_input]})
        return prediction
    with tf.Session(graph=tf.Graph()) as sess:
        tf.saved_model.loader.load(
            sess,
            [tf.saved_model.tag_constants.SERVING],
            export_dir
        )
        for filename in filenames:
            with open(filename) as fp:
                for line in fp:
                    # because of the batch set, result[i] is the ith prediction
                    result = predict(line, sess)
                    print(result[0])  # eg. [0.84468544]

def main(model_dir, mode="export_model_test"):
    filenames = ["data/kdda.t"]
    if "export_model_test" == mode:
        export_model_predict(model_dir, filenames)


if __name__ == "__main__":
    print("main start")
    model_dir = "./export_model/1546424071/"
    main(model_dir, "export_model_test")
    print("main finished")

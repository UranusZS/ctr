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


class SparseFactorizationMachine(object):
    def __init__(self, logger, model_name="sparse_fm"):
        self.model_name = model_name
        self.logger = logger 

    def build(self, features, labels, mode, params):
        batch_size = params["batch_size"]
        feature_max_num = params["feature_max_num"]
        optimizer_type = params["optimizer_type"]
        factor_vec_size = params["factor_size"]
        
        logger.info("export features {0}".format(features))
        logger.info("mode {0}".format(mode))
        logger.info("params {0}".format(params))
        
        if mode == tf.estimator.ModeKeys.PREDICT:
            """
            # for export
            sp_indexes = tf.SparseTensor(indices=features['feature_id_indices'],
                                         values=features['feature_id_values'],
                                         dense_shape=features['feature_id_shapes'])
            sp_vals = tf.SparseTensor(indices=features['feature_val_indices'],
                                      values=features['feature_val_values'],
                                      dense_shape=features['feature_val_shapes'])
            """
            # for input_fn predict
            sp_indexes = features['feature_ids']
            sp_vals = features['feature_vals']
            logger.info("sp: {0}, {1}".format(sp_indexes, sp_vals))
            
        if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
            sp_indexes = features['feature_ids']
            sp_vals = features['feature_vals']
            logger.info("sp: {0}, {1}".format(sp_indexes, sp_vals))
        
        bias = tf.get_variable(name="b", shape=[1], initializer=tf.glorot_normal_initializer())
        w_first_order = tf.get_variable(name='w_first_order', shape=[feature_max_num, 1], initializer=tf.glorot_normal_initializer())
        linear_part = tf.nn.embedding_lookup_sparse(w_first_order, sp_indexes, sp_vals, combiner="sum") + bias
        
        w_second_order = tf.get_variable(name='w_second_order', shape=[feature_max_num, factor_vec_size], initializer=tf.glorot_normal_initializer())
        embedding = tf.nn.embedding_lookup_sparse(w_second_order, sp_indexes, sp_vals, combiner="sum")
        embedding_square = tf.nn.embedding_lookup_sparse(tf.square(w_second_order), sp_indexes, tf.square(sp_vals), combiner="sum")
        sum_square = tf.square(embedding)
        second_part = 0.5*tf.reduce_sum(tf.subtract(sum_square, embedding_square), 1)
        
        y_hat = linear_part + tf.expand_dims(second_part, -1)
        predictions = tf.sigmoid(y_hat)
        logger.info("y_hat: {0}, second_part: {1}, linear_part: {2}".format(y_hat, second_part, linear_part))
        pred = {
            "label": features["label"],
            "prob": predictions
        }
        export_outputs = {
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf.estimator.export.PredictOutput(predictions)
        }
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(
                mode=mode, 
                predictions=pred,
                export_outputs=export_outputs)
        
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=tf.squeeze(y_hat)))
        tf.summary.scalar("loss", loss)
        if optimizer_type == "sgd":
            opt = tf.train.GradientDescentOptimizer(learning_rate=params['learning_rate'])
        elif optimizer_type == "ftrl":
            opt = tf.train.FtrlOptimizer(learning_rate=params['learning_rate'],)
        elif optimizer_type == "adam":
            opt = tf.train.AdamOptimizer(learning_rate=params['learning_rate'])
        elif optimizer_type == "momentum":
            opt = tf.train.MomentumOptimizer(learning_rate=params['learning_rate'], momentum=params['momentum'])
            
        train_step = opt.minimize(loss,global_step=tf.train.get_global_step())
        auc = tf.metrics.auc(labels, predictions)
        eval_metric_ops = {
            "auc" : auc 
        }
        tf.summary.scalar("auc", auc[1])

        if mode == tf.estimator.ModeKeys.TRAIN:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, loss=loss, train_op=train_step, eval_metric_ops=eval_metric_ops)
        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, loss=loss, eval_metric_ops=eval_metric_ops)


def export_model(classifier, export_dir="./export_model"):
    """
    export_model
    """
    # for exporting saved model
    feature_spec = {
        "feature_id_indices": tf.placeholder(dtype=tf.int64, name='feature_ids/indices'),
        "feature_id_values": tf.placeholder(dtype=tf.int64, name='feature_ids/values'),
        "feature_id_shapes": tf.placeholder(dtype=tf.int64, name='feature_ids/shape'),
        "feature_val_indices": tf.placeholder(dtype=tf.int64, name='feature_vals/indices'),
        "feature_val_values": tf.placeholder(dtype=tf.float32, name='feature_vals/values'),
        "feature_val_shapes": tf.placeholder(dtype=tf.int64, name='feature_vals/shape'),
    }
    feature_spec = {
            "label": tf.FixedLenFeature([], tf.float32),
            "feature_ids": tf.VarLenFeature(tf.int64),
            "feature_vals": tf.VarLenFeature(tf.float32)
    }
    #serving_input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(
    #        feature_spec)
    serving_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec, 1)
    classifier.export_savedmodel(
            export_dir,
            serving_input_fn)


def export_parsing_model(classifier, export_dir="./export_model"):
    """
    export_model
    """
    # for exporting saved model
    feature_spec = {
            "label": tf.FixedLenFeature([], tf.float32),
            "feature_ids": tf.VarLenFeature(tf.int64),
            "feature_vals": tf.VarLenFeature(tf.float32)
    }
    serving_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec, 1)
    classifier.export_savedmodel(
            export_dir,
            serving_input_fn)


def get_params():
    params = {
        "batch_size": 32,
        "feature_max_num": 5,
        "optimizer_type": "sgd",
        "factor_size": 128,
        "learning_rate": 0.001,
        "steps": 20,
        #"steps": 20000000,
    }
    return params


def train(filenames, model_dir):
    model_obj = SparseFactorizationMachine(logger)
    print(model_obj)
    params = get_params()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    run_config = tf.estimator.RunConfig(
                    model_dir=model_dir, 
                    save_checkpoints_steps=10000, 
                    keep_checkpoint_max=5)

    classifier = tf.estimator.Estimator(
        model_fn=model_obj.build,
        config=run_config,
        params=params
    )
    print(classifier)
    classifier.train(
        input_fn=lambda :tfrecord_input_fn(filenames, shuffle=True, 
                                 batch_size=params.get("batch_size", 32), 
                                 num_epochs=2),
        steps=params.get("steps", 20000))
    export_model(classifier)

def predict(filenames, model_dir):
    model_obj = SparseFactorizationMachine(logger)
    print(model_obj)
    params = get_params()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    run_config = tf.estimator.RunConfig(
                    model_dir=model_dir, 
                    save_checkpoints_steps=10000, 
                    keep_checkpoint_max=5)
    classifier = tf.estimator.Estimator(
        model_fn=model_obj.build,
        config=run_config,
        params=params
    )
    print(classifier)
    predicts = classifier.predict(
        input_fn=lambda :tfrecord_input_fn(filenames, shuffle=False, 
                                 batch_size=params.get("batch_size", 32), 
                                 num_epochs=1)
        )
    with open("./data/predict_result", "w") as fp:
        for item in predicts:
            print(item)
            out_str = ""
            out_str += "\t" + str(item['label'])
            out_str += "\t" + str(item["prob"][0])
            fp.write(out_str + "\n")


def main(model_dir, mode="train"):
    train_filenames = ["./data/kdda_tfrecord/kdda_00001.tfrecord"]
    predict_filenames = ["./data/kdda_tfrecord/kdda_00001.tfrecord"]
    if "train" == mode:
        train(train_filenames, model_dir)
    if "predict" == mode:
        predict(predict_filenames, model_dir)

if __name__ == "__main__":
    print("main start")
    model_dir = "./model/"
    #main(model_dir, "predict")
    main(model_dir, "train")
    print("main finished")

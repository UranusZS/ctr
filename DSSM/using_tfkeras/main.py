# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import sys
import six
import json
import math
import argparse
import traceback
import importlib
import numpy as np

import tensorflow as tf
from tensorflow import keras

from dssm_reader import dataset_reader
from dssm_reader import parser_raw_line
from dssm_simple import create_model, create_metrics, create_loss
import dssm_config
from file_utils import *


def clear_model_dir(args):
    clear_dir = [args.model_path, args.checkpoint_path, args.tensorboard_log]
    for d in clear_dir:
        if d and fs_exist(d):
            fs_remove(d)
    # ./data/model/keras//DSSM/main
    fs_mkdir(args.model_path)


def parse_args():
    parser = argparse.ArgumentParser(description="dssm args parser")
    parser.add_argument('--name', dest='model_name',
                      help='the name of the model',
                      default='dssm', type=str)
    parser.add_argument('--train', dest='train_dir',
                      help='the dir of the train files',
                      default='./data/train/', type=str)
    parser.add_argument('--test', dest='test_dir',
                      help='the dir of the test files',
                      default='./data/test/', type=str)
    parser.add_argument('--mode', dest='mode',
                      choices = ["TRAIN", "PREDICT", "EVAL", "QUERY_EMB", "DOC_EMB",],
                      help='mode of the main process',
                      default='TRAIN', type=str)
    parser.add_argument('--model', dest='model_path',
                      help='the dir of the model',
                      default='./data/model/keras/', type=str)
    parser.add_argument('--checkpoint', dest='checkpoint_path',
                      help='the dir of the model checkpoint',
                      default='./data/checkpoint/', type=str)
    parser.add_argument('--tensorboard', dest='tensorboard_log',
                      help='the dir of the model tensorboard',
                      default='./data/tensorboard/', type=str)
    parser.add_argument('--pretrain', dest='pretrain',
                      choices = ["No", "Yes"],
                      help='whether to use pretrained model',
                      default='No', type=str)
    if len(sys.argv) == 1:
        parser.print_help()
    FLAGS, unparsed = parser.parse_known_args()
    FLAGS.model_path = "{}/{}".format(FLAGS.model_path, FLAGS.model_name)
    FLAGS.checkpoint_path = "{}/{}/{}".format(FLAGS.checkpoint_path, FLAGS.model_name, FLAGS.model_name)
    FLAGS.tensorboard_log = "{}/{}".format(FLAGS.tensorboard_log, FLAGS.model_name)
    FLAGS.pretrain = False
    if FLAGS.pretrain == "Yes":
        FLAGS.pretrain = True
    return FLAGS


def save_model(model, query_model, doc_model, keras_model_path):
    tf.keras.models.save_model(model, "{}/{}".format(keras_model_path, "main.h5"), 
                            overwrite=True, include_optimizer=True)
    tf.keras.models.save_model(query_model, "{}/{}".format(keras_model_path, "query.h5"), 
                            overwrite=True, include_optimizer=True)
    tf.keras.models.save_model(doc_model, "{}/{}".format(keras_model_path, "doc.h5"), 
                            overwrite=True, include_optimizer=True)


def load_model(model, keras_model_path, sub_model_name):
    model_dir = "{}/{}".format(keras_model_path, sub_model_name)
    return model.load_weights(model_dir) 


def train(args, config):
    print("{} begins {}".format(args.mode, "-"*10))
    if not args.pretrain:
        clear_model_dir(args)
    print("model config is {}".format(json.dumps(config)))
    # define and compile model
    # TODO distributed training
    mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"])
    # with mirrored_strategy.scope():
    if True:
        model, query_model, doc_model = create_model(**config)
        optimizer = tf.keras.optimizers.Adam(lr=0.02, beta_1=0.9, beta_2=0.999,
                        epsilon=None, decay=0.999, amsgrad=False)
        metrics = create_metrics(**config)
        loss = create_loss(**config)
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    # train model
    train_files = fs_list(args.train_dir)[0]
    dataset = dataset_reader(train_files, shuffle=True, batch_size=128, repeat_num=200)
    # define train callbacks
    cp_callback = tf.keras.callbacks.ModelCheckpoint(args.checkpoint_path,
                                                      save_weights_only=True,
                                                      verbose=1)
    tensorboard_cbk = tf.keras.callbacks.TensorBoard(log_dir=args.tensorboard_log)
    callbacks = [cp_callback, tensorboard_cbk]
    epochs = 2
    steps_per_epoch = 256
    history = model.fit(dataset, epochs=epochs, steps_per_epoch=steps_per_epoch, 
                          callbacks = callbacks)
    save_model(model, query_model, doc_model, args.model_path)
    print("{} ends   {}".format(args.mode, "-"*10))
    pass 


def softmax(z):
    assert len(z.shape) == 2
    s = np.max(z, axis=1)
    s = s[:, np.newaxis] # necessary step to do broadcasting
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis] # dito
    return e_x / div


def sub_model_test(args, config):
    model, query_model, doc_model = create_model(**config)
    load_model(model, args.model_path, "main.h5")
    load_model(query_model, args.model_path, "query.h5")
    load_model(doc_model, args.model_path, "doc.h5")
    test_files = fs_list(args.test_dir)[0]
    dataset = dataset_reader(test_files, shuffle=False, batch_size=128, repeat_num=1)
    dataset_iter = iter(dataset)
    try:
        while True:
            print("*"*10)
            features, label = dataset_iter.next()
            batch = []
            batch.append(features['query_Input'])
            for i in range(config.get("doc_num", 5)):
                batch.append(features['doc{}_Input'.format(i)])
            score_result = model.predict_on_batch(batch)
            for res in score_result:
                print("\t".join(["{}".format(r) for r in res]))
            query_emb = query_model.predict_on_batch(batch[0])
            query_emb_n = np.linalg.norm(query_emb, axis=1, keepdims=True)
            # make sure to normalize the embedding
            query_emb = query_emb / query_emb_n
            batch_size = query_emb.shape[0]
            distance_arr = np.empty(shape=[0, batch_size], dtype=np.float64)
            for b in batch[1:]:
                doc_emb = doc_model.predict_on_batch(b)
                doc_emb_n = np.linalg.norm(doc_emb, axis=1, keepdims=True)
                # make sure to normalize the embedding
                doc_emb = doc_emb / doc_emb_n
                distance_mat = query_emb.dot(doc_emb.transpose((1, 0))) 
                distances = np.diagonal(distance_mat)
                distances = np.expand_dims(distances, axis=0)
                distance_arr = np.concatenate((distance_arr, distances), axis=0)
            print(softmax(distance_arr.T))
            print("*"*10)
    except:
        traceback.print_exc()
        pass
    pass 



def predict(args, config):
    print("{} begins {}".format(args.mode, "-"*10))
    model, query_model, doc_model = create_model(**config)
    load_model(model, args.model_path, "main.h5")
    test_files = fs_list(args.test_dir)[0]
    dataset = dataset_reader(test_files, shuffle=False, batch_size=128, repeat_num=1)
    dataset_iter = iter(dataset)
    try:
        while True:
            features, label = dataset_iter.next()
            batch = []
            batch.append(features['query_Input'])
            for i in range(config.get("doc_num", 5)):
                batch.append(features['doc{}_Input'.format(i)])
            result = model.predict_on_batch(batch)
            for res in result:
                print("\t".join(["{}".format(r) for r in res]))
    except:
        #traceback.print_exc()
        pass

    print("{} ends   {}".format(args.mode, "-"*10))
    pass 


def query_emb(args, config):
    print("{} begins {}".format(args.mode, "-"*10))
    model, query_model, doc_model = create_model(**config)
    load_model(query_model, args.model_path, "query.h5")
    test_files = fs_list(args.test_dir)[0]
    dataset = dataset_reader(test_files, shuffle=False, batch_size=128, repeat_num=1)
    dataset_iter = iter(dataset)
    try:
        while True:
            features, label = dataset_iter.next()
            batch = []
            batch.append(features['query_Input'])
            result = query_model.predict_on_batch(batch)
            # make sure to normalize the embedding
            norm = np.linalg.norm(result, axis=1, keepdims=True)
            #print(norm)
            result = result / norm
            for res in result:
                print("\t".join(["{}".format(r) for r in res]))
                pass
    except:
        traceback.print_exc()
        pass
    print("{} ends   {}".format(args.mode, "-"*10))
    pass 

def query_emb_raw(args, config):
    print("{} begins {}".format(args.mode, "-"*10))
    model, query_model, doc_model = create_model(**config)
    load_model(query_model, args.model_path, "query.h5")
    test_files = fs_list(args.test_dir)[0]
    for filename in test_files:
        with open(filename) as fp:
            for line in fp:
                features, label = parser_raw_line(line)
                #print(json.dumps(features))
                query_input = features["query_Input"]
                emb = query_model.predict_on_batch([[query_input]])
                # TODO normalize the result 
                print("\t".join([str(x) for x in emb[0].tolist()]))
    print("{} ends   {}".format(args.mode, "-"*10))
    pass 


def doc_emb(args, config):
    print("{} begins {}".format(args.mode, "-"*10))
    model, query_model, doc_model = create_model(**config)
    load_model(doc_model, args.model_path, "doc.h5")
    test_files = fs_list(args.test_dir)[0]
    dataset = dataset_reader(test_files, shuffle=False, batch_size=128, repeat_num=1)
    dataset_iter = iter(dataset)
    try:
        while True:
            features, label = dataset_iter.next()
            batch = []
            batch.append(features['doc0_Input'])
            result = doc_model.predict_on_batch(batch)
            # make sure to normalize the embedding
            norm = np.linalg.norm(result, axis=1, keepdims=True)
            result = result / norm
            for res in result:
                print("\t".join(["{}".format(r) for r in res]))
    except:
        #traceback.print_exc()
        pass
    print("{} ends   {}".format(args.mode, "-"*10))
    pass 


def doc_emb_raw(args, config):
    print("{} begins {}".format(args.mode, "-"*10))
    model, query_model, doc_model = create_model(**config)
    load_model(doc_model, args.model_path, "doc.h5")
    test_files = fs_list(args.test_dir)[0]
    for filename in test_files:
        with open(filename) as fp:
            for line in fp:
                features, label = parser_raw_line(line)
                #print(json.dumps(features))
                doc_input = features["doc0_Input"]
                emb = doc_model.predict_on_batch([[doc_input]])
                # TODO normalize the result 
                print("\t".join([str(x) for x in emb[0].tolist()]))
    print("{} ends   {}".format(args.mode, "-"*10))
    pass 


def evaluate(args, config):
    print("{} begins {}".format(args.mode, "-"*10))
    print("{} ends   {}".format(args.mode, "-"*10))
    pass


def main():
    print("="*20)
    args = parse_args() 
    args_dict = vars(args)
    config = dssm_config.config
    config.update(args_dict)
    print("="*20)
    if args.mode == "TRAIN":
        train(args, config)
    if args.mode == "PREDICT":
        predict(args, config)
    if args.mode == "EVAL":
        evaluate(args, config)
    if args.mode == "QUERY_EMB":
        query_emb(args, config)
    if args.mode == "DOC_EMB":
        doc_emb(args, config)
    print("="*20)
    pass


if __name__ == "__main__":
    main()

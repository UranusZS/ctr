# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import sys
import json
import math
import argparse
import traceback
import numpy as np

import tensorflow as tf
from tensorflow import keras 

L = tf.keras.layers

def create_loss(**kwargs):
    return tf.keras.losses.CategoricalCrossentropy()

def create_model(**kwargs):
    # default configs
    query_len = 15
    doc_len = 15
    doc_num = 5
    vocab_size = 1005444
    emb_size = 64
    mlp_sizes = [300, 256, 256, 128]
    conv_max_sizes = [[3, 128, 4], [4, 128, 3]]
    # get config from kwargs
    if kwargs and "query_len" in kwargs:
        query_len = kwargs.get("query_len", query_len)
    if kwargs and "doc_len" in kwargs:
        doc_len = kwargs.get("doc_len", doc_len)
    if kwargs and "doc_num" in kwargs:
        doc_num = kwargs.get("doc_num", doc_num)
    if kwargs and "vocab_size" in kwargs:
        vocab_size = kwargs.get("vocab_size", vocab_size)
    if kwargs and "emb_size" in kwargs:
        emb_size = kwargs.get("emb_size", emb_size)
    if kwargs and "vocab_size" in kwargs:
        vocab_size = kwargs.get("vocab_size", vocab_size)
    if kwargs and "emb_size" in kwargs:
        emb_size = kwargs.get("emb_size", emb_size)
    embedding_activation = tf.nn.relu
    activation = tf.nn.relu
    # 注意不同尺寸的filter在concat的时候，因为输出维度不同，需要调整max_pool的大小，以便对应维度相同
    def get_conv_pool_layers(conv_max_sizes):
        _convs = []
        for i, size in enumerate(conv_max_sizes):
            out_filters = size[1]
            conv_size = size[0]
            pool_size = size[2]
            conv = L.Conv1D(out_filters, conv_size)   # (batch, steps, features)
            bn = L.BatchNormalization()
            max_pool = L.GlobalMaxPool1D()             # (batch_size, features)
            avg_pool = L.GlobalAveragePooling1D()
            _convs.append([conv, bn, max_pool, avg_pool])
        return _convs
    def get_mlps(mlp_sizes, prefix=""):
        _mlps = []
        for i, mlp_size in enumerate(mlp_sizes):
            if prefix:
                mlp = L.Dense(mlp_size, activation=activation, name="{}_{}".format(prefix, i))
            else:
                mlp = L.Dense(mlp_size, activation=activation)
            _mlps.append(mlp)
        return _mlps

    def get_embeding_model(query_len, conv_max_sizes, mlp_sizes):
        query_input = tf.keras.Input(shape=(query_len,))   # (batch_size, query_len)
        # define layers
        query_emb = L.Embedding(vocab_size, emb_size,
                                input_length=query_len)  # (batch_size, query_len, emb_size)
        query_convs = get_conv_pool_layers(conv_max_sizes)
        query_mlps = get_mlps(mlp_sizes)
        # define graph
        query = query_emb(query_input)
        query_concate = []
        for i, layers in enumerate(query_convs):
            x = layers[0](query)
            x = layers[1](x)
            y = layers[2](x)
            z = layers[3](x)
            x = L.concatenate([y, z], axis=1)
            query_concate.append(x)
        query = L.concatenate(query_concate, axis=1)
        for i, layer in enumerate(query_mlps):
            query = layer(query)
        return tf.keras.Model(query_input, query)

    # define input layers
    query_input = tf.keras.Input(shape=(query_len,), name = "query_Input")   # (batch_size, query_len)
    doc_inputs = [tf.keras.Input(shape=(doc_len,), name="doc{}_Input".format(i)) for i in range(doc_num)]
    # define submodels
    query_model = get_embeding_model(query_len, conv_max_sizes, mlp_sizes)
    doc_model = get_embeding_model(doc_len, conv_max_sizes, mlp_sizes)

    query = query_model(query_input)
    docs = []
    for i, doc_input in enumerate(doc_inputs):
        doc_emb = doc_model(doc_input)
        docs.append(doc_emb)
    # define distance
    R = [L.dot([query, doc], axes=1, normalize=True) for doc in docs]
    distance = L.concatenate(R)
    prob = L.Activation(activation=tf.nn.softmax)(distance)

    model = tf.keras.Model(inputs=[query_input, doc_inputs], outputs=prob)
    return model, query_model, doc_model


def create_metrics(**kwargs):
    metrics=[tf.keras.metrics.CategoricalAccuracy(), "mae", "mse", tf.keras.metrics.CategoricalCrossentropy(), tf.keras.metrics.kld,]
    return metrics 


if __name__ == "__main__":
    print("This is {}".format(__file__))
    model, loss, query_model, doc_model = create_model()
    model.summary()
    query_model.summary()
    doc_model.summary()
    tf.keras.utils.plot_model(model, 'model_test.png', show_shapes=True
)
        


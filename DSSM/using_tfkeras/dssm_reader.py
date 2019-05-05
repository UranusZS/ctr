# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import sys
import json
import argparse
import numpy as np

import tensorflow as tf


def parser_raw_line(line, col_sep=";", val_sep=",", **kwargs):
    doc_len = 15
    doc_num = 5
    if kwargs and "doc_len" in kwargs:
        doc_len = kwargs.get("doc_len", doc_len)
    if kwargs and "doc_num" in kwargs:
        doc_num = kwargs.get("doc_num", doc_num)
    line_arr = line.strip().split(col_sep)
    _id = line_arr[0]
    query_strs = line_arr[1].strip().split(val_sep)
    label_strs = line_arr[2].strip().split(val_sep)
    doc_strs = line_arr[3].strip().split(val_sep)
    features = {
        "_id": _id,
        "query_Input": [int(x) for x in query_strs],
    }
    for i in range(doc_num):
        start = i * doc_len
        features["doc{}_Input".format(i)] = [int(x) for x in doc_strs[start: start+doc_len]] 
    return features, [float(x) for x in label_strs]


def _line_parser_dssm(line, col_sep=";", val_sep=",", **kwargs):
    """
    _parse_line
        lineid;query;labels;docs
        l1;1,2,3;1,0,0,0;1,2,3,4,5,6,7,8,9,10,11,12
    """
    doc_len = 15
    doc_num = 5
    if kwargs and "doc_len" in kwargs:
        doc_len = kwargs.get("doc_len", doc_len)
    if kwargs and "doc_num" in kwargs:
        doc_num = kwargs.get("doc_num", doc_num)
    line_arr = tf.strings.split([line], col_sep).values
    _id = line_arr[0]
    query_str = tf.strings.split([line_arr[1]], val_sep).values
    query = tf.strings.to_number(query_str, tf.int32)
    label_str = tf.strings.split([line_arr[2]], val_sep).values
    label = tf.strings.to_number(label_str, tf.float32)
    docs_str = tf.strings.split([line_arr[3]], val_sep).values
    docs = tf.strings.to_number(docs_str, tf.int32)
    doc_lens = [doc_len] * doc_num
    docs = tf.split(docs, num_or_size_splits=doc_lens, axis=0)
    features = {}
    features["_id"] = _id 
    features["query_Input"] = query
    for i, doc in enumerate(docs):
        features["doc{}_Input".format(i)] = doc
    return features, label

def dataset_reader(filenames, shuffle=False, batch_size=128,
        repeat_num=1, line_parser=_line_parser_dssm, col_sep=";", val_sep=","):
    """
        dataset reader
    """
    #dataset = tf.data.TextLineDataset(filenames)
    filenames_dataset = tf.data.Dataset.from_tensor_slices(filenames)
    dataset = filenames_dataset.flat_map(
            lambda filename: (
                tf.data.TextLineDataset(filename)
            )
        )
    if shuffle:
        dataset = dataset.shuffle(1280)
    dataset = dataset.map(lambda x: line_parser(x, col_sep, val_sep), num_parallel_calls=3)
    dataset = dataset.repeat(repeat_num).prefetch(8*batch_size).batch(batch_size)
    #iterator = dataset.make_one_shot_iterator()
    #return iterator.get_next()
    return dataset


if __name__ == '__main__':
    print("This is {}".format(__file__))


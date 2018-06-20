#coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import json
import shutil
import argparse

import tensorflow as tf

_separator = "/"
fc_separator = "#"
_inner_prefix = "fc"


def numeric_column(fc_source, fc_options):
    shape = fc_options.get("shape", None)
    fc_source = fc_source[0]
    if shape is not None and not isinstance(shape, list):
        shape = [shape]
    if shape is not None:
        return tf.feature_column.numeric_column(key=fc_source, shape=shape),
    shape
    return tf.feature_column.numeric_column(key=fc_source), [1]

def bucketized_column(fc_source, fc_options):
    boundaries = fc_options.get("boundaries", None)
    fc_source = fc_source[0]
    if isinstance(fc_source, str):
        # First, convert the raw input to a numeric column.
        fc_source = tf.feature_column.numeric_column(fc_source)
    return tf.feature_column.bucketized_column(
            source_column = fc_source,
            boundaries = boundaries), [len(boundaries) + 1]

def categorical_column_with_identity(fc_source, fc_options):
    num_buckets = fc_options.get("num_buckets", 1)
    fc_source = fc_source[0]
    return tf.feature_column.categorical_column_with_identity(
            key=fc_source,
            num_buckets=num_buckets), [num_buckets]

def categorical_column_with_vocabulary_list(fc_source, fc_options):
    vocabulary_list = fc_options.get("vocabulary_list", [])
    num_oov_buckets = fc_options.get("num_oov_buckets", None)
    fc_source = fc_source[0]
    if num_oov_buckets is None: 
        return tf.feature_column.categorical_column_with_vocabulary_list(
                key=fc_source,
                vocabulary_list=vocabulary_list), [len(vocabulary_list)]
    return tf.feature_column.categorical_column_with_vocabulary_list(
            key=fc_source,
            num_oov_buckets=num_oov_buckets,
            vocabulary_list=vocabulary_list), [len(vocabulary_list) +
                    num_oov_buckets]

def categorical_column_with_vocabulary_file(fc_source, fc_options):
    vocabulary_file = fc_options.get("vocabulary_file", "vocab.txt")
    vocabulary_size = fc_options.get("vocabulary_size", 3)
    num_oov_buckets = fc_options.get("num_oov_buckets", 5)
    fc_source = fc_source[0]
    return tf.feature_column.categorical_column_with_vocabulary_file(
            key=fc_source,
            vocabulary_file=vocabulary_file,
            vocabulary_size=vocabulary_size), [vocabulary_size]

def categorical_column_with_hash_bucket(fc_source, fc_options):
    hash_bucket_size = fc_options.get("hash_bucket_size", 3)
    fc_source = fc_source[0]
    return tf.feature_column.categorical_column_with_hash_bucket(
            key=fc_source,
            hash_bucket_size=hash_bucket_size), [hash_bucket_size]

def crossed_column(fc_source, fc_options):
    hash_bucket_size = fc_options.get("hash_bucket_size", 3)
    print(fc_source)
    return tf.feature_column.crossed_column(
            keys=fc_source,
            hash_bucket_size=hash_bucket_size), [hash_bucket_size]

def indicator_column(fc_source, fc_options):
    # maybe a size parameter is prefered
    fc_source_size = fc_options.get("fc_source_size", 0)
    fc_source = fc_source[0]
    return tf.feature_column.indicator_column(
            categorical_column=fc_source), [fc_source_size]

def embedding_column(fc_source, fc_options):
    dimension = fc_options.get("dimension", 8)
    fc_source = fc_source[0]
    return tf.feature_column.embedding_column(
            categorical_column=fc_source,
            dimension=dimension), [dimension]

_fc_function = {
        "numeric_column": numeric_column,
        "bucketized_column": bucketized_column,
        "categorical_column_with_identity": categorical_column_with_identity,
        "categorical_column_with_vocabulary_list":
        categorical_column_with_vocabulary_list,
        "categorical_column_with_vocabulary_file":
        categorical_column_with_vocabulary_file,
        "categorical_column_with_hash_bucket":
        categorical_column_with_hash_bucket,
        "crossed_column": crossed_column,
        "indicator_column": indicator_column,
        "embedding_column": embedding_column
    }

def read_feature_columns_cfg(filename="feature_columns.cfg"):
    """
    feature_columns.cfg
    """
    fc_raw_dict = {}
    with open(filename) as fp:
        head = fp.readline()
        line = fp.readline()
        while line:
            # parse the config line by line
            fc_group, fc_source, fc_name, fc_type, fc_options = \
                                                    line.strip().split("\t")
            try:
                fc_options = json.loads(fc_options)
            except:
                fc_options = {}
            name = _separator.join([_inner_prefix, fc_group, fc_name])
            fc_raw_dict[name] = [fc_group, fc_source, fc_type, fc_options]
            line = fp.readline()
    return fc_raw_dict

def read_columns_cfg(filename="columns.txt"):
    """
    read_columns_cfg
    """
    _columns = []
    _columns_default = []
    with open(filename) as fp:
        for line in fp.readlines():
            line_arr = line.strip().split("\t")
            _columns.append(line_arr[0].strip())
            # gen default value
            if 'str' == line_arr[1].strip():
                if len(line_arr) > 2:
                    _columns_default.append([str(line_arr[2])])
                else:
                    _columns_default.append([''])
            if 'int' == line_arr[1].strip():
                if len(line_arr) > 2:
                    _columns_default.append([int(line_arr[2])])
                else:
                    _columns_default.append([0])
    return _columns, _columns_default


def build_feature_columns(fc_cfg_file="feature_columns.cfg"):
    """
    build_feature_columns
    """
    feature_columns_dict = {}
    #columns, columns_default = read_columns_cfg(column_file)
    fc_raw_dict = read_feature_columns_cfg(fc_cfg_file)

    def build_column(key, row_cfg, res_dict):
        """
        build_column
        """
        print("------ start building {0} ------".format(key))
        print(res_dict)
        fc_group = row_cfg[key][0]
        fc_source = row_cfg[key][1]
        fc_type = row_cfg[key][2]
        fc_options = row_cfg[key][3]
        # fc_raw_dict[name] = [fc_group, fc_source, fc_type, fc_options]
        fc_source_list = fc_source.split(fc_separator)
        src_column_list = []
        # maybe several fc_source_columns needed
        for source in fc_source_list:
            if source.startswith(_inner_prefix):
                _, group, name = source.split(_separator)
                # if source have not yet built, built it recursively
                if source not in res_dict[group]:
                    print("        ++++++ building sub {0} ++++++".format(source))
                    build_column(source, row_cfg, res_dict)
                src_column_list.append(res_dict.get(group).get(source)[0])
            else:
                src_column_list.append(source)
            if "indicator_column" == fc_type:
                fc_options["fc_source_size"] = res_dict.get(group).get(source)[1][0]
        # every dependency is ready, start to build 
        print(fc_options)
        fc_column = _fc_function.get(fc_type)(src_column_list, fc_options)
        res_dict[fc_group][key] = fc_column
        return

    print(json.dumps(fc_raw_dict))
    for (key, value) in fc_raw_dict.items():
        fc_group = value[0]
        # check if group exists, add empty dict if not
        if fc_group not in feature_columns_dict:
            feature_columns_dict[fc_group] = {}
        # if feature column not built, build it
        if key not in feature_columns_dict[fc_group]:
            build_column(key, fc_raw_dict, feature_columns_dict)

    print(feature_columns_dict)
    print(feature_columns_dict["deep"].values())
    return feature_columns_dict

        
def get_feature_columns(feature_columns_dict, keys=["wide", "deep"]):
    """
    get_feature_columns 
    """
    result = []
    for key in keys:
        result.append([x[0] for x in feature_columns_dict[key].values()])
    return result


if __name__ ==  "__main__":
    """
    cfg_dict = read_feature_columns_cfg()
    print(cfg_dict)
    """
    build_feature_columns()


# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import sys
import time
import json
import random
import datetime

import hashlib
import argparse

from pyspark import SparkContext
from pyspark.sql import SQLContext, Row
from operator import add

reload(sys)
sys.setdefaultencoding('utf8')


def main(sc=None, input_dir=None, output_dir=None):
    """
    main exec
    """
    print("input_dir {0} output_dir {1}".format(input_dir, output_dir))
    key_separator = "_"

    inputs = sc.textFile(input_dir, minPartitions=100, use_unicode=False)
    #print(inputs.take(5))

    def parse_inputs(line, separator="\t"):
        """
        parse log 
        """
        log_dict = json.loads(line.strip())
        advertiserid = log_dict.get("advertiserid", "")
        campaignid = log_dict.get("campaignid", "")
        solutionid = log_dict.get("solutionid", "")
        creativeid = log_dict.get("creativeid", "")
        is_click = int(log_dict.get("is_click", 0))
        materialid = log_dict.get("materialid", "1")

        key = key_separator.join([advertiserid, campaignid, solutionid,
                str(creativeid), materialid])
        value = [1, is_click]
        return (key, value)


    inputs = inputs.map(lambda x: parse_inputs(x)).filter(lambda x: x[0] !=
            "NULL")

    def reduce_fun(a, b):
        return [a[0]+b[0], a[1]+b[1]]

    inputs = inputs.reduceByKey(reduce_fun)

    def output_format(x):
        out_arr = x[0].split("_") + [str(t) for t in x[1]]
        out_arr = [t.replace("\t", "") for t in out_arr]
        out_str = "\t".join(out_arr)
        return out_str

    inputs = inputs.map(lambda x: output_format(x))

    inputs = inputs.repartition(10)
    inputs.saveAsTextFile(output_dir)
    #inputs.saveAsTextFile(output_dir, "org.apache.hadoop.io.compress.GzipCodec")
    print("main finished")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="argparser")
    parser.add_argument("--input_dir", type=str, help="the input data path",
                            default="")
    parser.add_argument("--output_dir", type=str, help="the output data path",
                            default="")
    parser.add_argument("--inputdate", type=str, help="input date",
                            default="2018-09-27")
    FLAGS, unparsed = parser.parse_known_args()

    sc = SparkContext(appName="show_click_stat")
    main(sc, FLAGS.input_dir, 
            FLAGS.output_dir
	)
    sc.stop()


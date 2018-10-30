#!/usr/bin/python
# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import time
import argparse
sys.path.append("../lib/")
from bayes_smoothing import BayesSmoothing 

def cal_pctr_by_bayes_smoothing(filein, fileout, alpha=1, beta=100, group_cid=2, imp_cid=5,
        click_cid=6):
    data = {}
    max_cid = max(group_cid, imp_cid, click_cid)
    with open(filein) as fp:
        for line in fp:
            line_arr = line.strip().split("\t")
            if len(line_arr) < max_cid:
                continue 
            key = line_arr[group_cid]
            impression = int(line_arr[imp_cid])
            click = int(line_arr[click_cid])
            if key not in data:
                data[key] = [[], []]
            data[key][0].append(impression)
            data[key][1].append(click)
    smooth_dict = {}
    for (k, v) in data.items():
        if len(v[0]) == 1:
            smooth_dict[k] = [alpha, beta]
        else:
            bayes = BayesSmoothing(alpha, beta)
            new_alpha, new_beta = bayes.fit(v[0], v[1])
            smooth_dict[k] = [new_alpha, new_beta]
    with open(fileout, "w") as ofp, open(filein) as ifp:
        for line in ifp:
            line_arr = line.strip().split("\t")
            if len(line_arr) < max_cid:
                continue 
            key = line_arr[group_cid]
            impression = int(line_arr[imp_cid])
            click = int(line_arr[click_cid])
            ctr = float(click) / (impression+0.00000001)
            pctr = BayesSmoothing(smooth_dict[key][0],
                    smooth_dict[key][1]).predict(impression, click)
            ctr = float("%.5f" % ctr)
            pctr = float("%.5f" % pctr)
            out_str = "\t".join([line.strip(), str(ctr), str(pctr)]) + "\n"
            ofp.write(out_str)


if __name__ == '__main__':
    localtime = time.asctime(time.localtime(time.time()))
    print("bayes_smoothing start: {0}".format(localtime))
    parser = argparse.ArgumentParser(description="argparser")
    parser.add_argument("--input", type=str, help="the input data path",
                            default="../data/20181028")
    parser.add_argument("--output", type=str, help="the output data path",
                            default="../data/result.20181028")
    FLAGS, unparsed = parser.parse_known_args()
    cal_pctr_by_bayes_smoothing(FLAGS.input, FLAGS.output)
    localtime = time.asctime(time.localtime(time.time()))
    print("bayes_smoothing ends: {0}".format(localtime))


# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

config = {
    "query_len": 15,
    "doc_len": 15,
    "doc_num": 5,
    "vocab_size": 1005444,
    "emb_size": 64,
    "conv_max_sizes": [[3, 128, 4], [4, 128, 3]],
    "mlp_sizes": [300, 256, 256, 128],
}


if __name__ == '__main__':
    import json 
    json.dumps(config)

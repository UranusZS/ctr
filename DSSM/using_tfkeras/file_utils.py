# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re 
import sys 
import shutil
import subprocess 
from enum import Enum


class FileSchema(Enum):
    HDFS = 1
    LOCAL = 2


def get_file_schema(filepath):
    if filepath.startswith("hdfs://"):
        return FileSchema.HDFS, filepath
    if filepath.startswith("file://"):
        return FileSchema.LOCAL, filepath[7:]
    return FileSchema.LOCAL, filepath


def put2hdfs(fname, hdfs_dir, hadoop_bin="hadoop"):
    cmd = hadoop_bin + " fs -put {0} {1}".format(fname, hdfs_dir)
    (status, output) = subprocess.getstatusoutput(cmd)
    return status


def list_hdfs(hdfs_dir, hadoop_bin="hadoop"):
    # or filelist=`${HADOOP_BIN} fs -ls ${input_base} | grep -v "^Found" | awk '{print $NF}' | sed ':label;N;s/\n/,/;b label' `
    cmd = hadoop_bin + " fs -ls {0} | grep -v items | {1}".format(hdfs_dir, "awk '{print $NF}' | sort")
    (status, output) = subprocess.getstatusoutput(cmd)
    if status != 0:
        return [], []
    hdfs_files = output.split()
    filenames = [x.split("/")[-1] for x in hdfs_files]
    return hdfs_files, filenames


def list_local_shell(local_dir):
    cmd = "ls {0}".format(local_dir)
    (status, output) = subprocess.getstatusoutput(cmd)
    if status != 0:
        return [], []
    filenames = output.split()
    local_files = ["{0}/{1}".format(local_dir, x) for x in filenames]
    return local_files, filenames


def list_local(local_dir):
    if not os.path.isdir(local_dir):
        if os.path.isfile(local_dir):
            return [local_dir], [local_dir.split("/")[-1]]
        return [], []
    filenames = os.listdir(local_dir)
    local_files = ["{0}/{1}".format(local_dir, x) for x in filenames]
    return local_files, filenames


def mkdir_hdfs(hdfs_dir, hadoop_bin="hadoop"):
    cmd = "{0} fs -mkdir {1}".format(hadoop_bin, hdfs_dir)
    (status, output) = subprocess.getstatusoutput(cmd)
    return status

def mkdir_local(local_dir):
    try:
        os.makedirs(local_dir)
    except:
        return -1
    return 0


def exist_hdfs(filename, hadoop_bin="hadoop"):
    cmd = "{0} fs -test -e {1}".format(hadoop_bin, filename)
    (status, output) = subprocess.getstatusoutput(cmd)
    if status != 0:
        return False
    return True


def exist_local(filename):
    return os.path.exists(filename)


def getsize_hdfs(filename, hadoop_bin="hadoop"):
    if not exist_hdfs(filename):
        return 0
    cmd = "{0} fs -dus {1} | {2}".format(hadoop_bin, filename, "awk '{print $2}'") 
    (status, output) = subprocess.getstatusoutput(cmd)
    return output

def getsize_local(filename):
    if not exist_local(filename):
        return 0
    return os.path.getsize(filename)


def get_linenum_hdfs(filename, hadoop_bin="hadoop"):
    '''this is slow'''
    if not exist_hdfs(filename):
        return 0
    cmd = "{0} fs -text {1} | wc -l".format(hadoop_bin, filename)
    (status, output) = subprocess.getstatusoutput(cmd)
    return int(output)


def get_linenum_local(filename):
    count = -1
    with open(filename) as fp:
        for count, line in enumerate(fp):
            pass
        count += 1
    return count


def remove_hdfs(filename, hadoop_bin="hadoop"):
    cmd = "{0} fs -rmr {1}".format(hadoop_bin, filename)
    (status, output) = subprocess.getstatusoutput(cmd)
    return status

def remove_local(filename):
    if not exist_local(filename):
        return -1
    if os.path.isdir(filename):
        shutil.rmtree(filename)
        return 0 
    os.remove(filename)
    return 0


def fs_list(filepath, hadoop_bin="hadoop"):
    schema, filename = get_file_schema(filepath)
    if schema == FileSchema.HDFS:
        return list_hdfs(filename, hadoop_bin)
    if schema == FileSchema.LOCAL:
        return list_local(filename)
    return NotImplemented


def fs_exist(filepath, hadoop_bin="hadoop"):
    schema, filename = get_file_schema(filepath)
    if schema == FileSchema.HDFS:
        return exist_hdfs(filename, hadoop_bin)
    if schema == FileSchema.LOCAL:
        return exist_local(filename)
    return NotImplemented


def fs_mkdir(filepath, hadoop_bin="hadoop"):
    schema, filename = get_file_schema(filepath)
    if schema == FileSchema.HDFS:
        return mkdir_hdfs(filename, hadoop_bin)
    if schema == FileSchema.LOCAL:
        return mkdir_local(filename)
    return NotImplemented


def fs_getsize(filepath, hadoop_bin="hadoop"):
    schema, filename = get_file_schema(filepath)
    if schema == FileSchema.HDFS:
        return getsize_hdfs(filename, hadoop_bin)
    if schema == FileSchema.LOCAL:
        return getsize_local(filename)
    return NotImplemented


def fs_remove(filepath, hadoop_bin="hadoop"):
    schema, filename = get_file_schema(filepath)
    if schema == FileSchema.HDFS:
        return remove_hdfs(filename, hadoop_bin)
    if schema == FileSchema.LOCAL:
        return remove_local(filename)
    return NotImplemented


def fs_get_linenum(filepath, hadoop_bin="hadoop"):
    schema, filename = get_file_schema(filepath)
    if schema == FileSchema.HDFS:
        return get_linenum_hdfs(filename, hadoop_bin)
    if schema == FileSchema.LOCAL:
        return get_linenum_local(filename)
    return NotImplemented


def test():
    filepath = "./file_utils.py"
    schema, filename = get_file_schema(filepath)
    print(filepath, schema == FileSchema.LOCAL, schema, filename)
    print(fs_exist(filepath))
    print(fs_list(filepath))
    print(fs_getsize(filepath))
    return


if __name__ == '__main__':
    test()

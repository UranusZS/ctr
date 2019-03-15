from __future__ import absolute_import, division, print_function

import os
import sys
import traceback
import numpy as np
import tensorflow as tf
from tensorflow import keras

tf.__version__
print(tf.VERSION)
print(tf.keras.__version__)

os.environ["CUDA_VISIBLE_DEVICES"]="0"

# input files
filenames = [
              "./data/train.data",
            ]
predict_filenames = ["./data/test.data"]
# check_point
checkpoint_path = "./model/training/lr.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
# keras model
keras_model_path = "./model/keras/lr.h5"

def gen_dataset(filenames, shuffle=False, batch_size=128, repeat_num=16):
    filename_dataset = tf.data.Dataset.from_tensor_slices(filenames)
    dataset = filename_dataset.flat_map(
                  lambda filename: (
                      tf.data.TextLineDataset(filename)
                  )
              )
    if shuffle:
        dataset = dataset.shuffle(128)

    def parser(line):
        line_arr = tf.string_split([line], "\t").values
        pid = line_arr[0]
        weight = tf.string_to_number(line_arr[1], tf.float32)
        label = tf.string_to_number(line_arr[2], tf.int32)
        feat = tf.string_split([line_arr[3]], ' ').values
        features = {}
        features['id'] = pid
        features['label'] = label
        features["feature"] = tf.string_to_number(feat, tf.float32)
        return features['feature'], label, weight

    dataset = dataset.map(parser, num_parallel_calls=3)
    dataset = dataset.repeat(repeat_num).batch(batch_size).prefetch(30)
    return dataset


class ScoreModel(object):

    def __init__(self):
        self._model = None
        self.create_model()
        return 

    def create_model(self):
        #  keras.layers.Dense(1, activation=tf.keras.activations.sigmoid, input_shape=(768,)),
        #])
        inputs = tf.keras.Input(shape=(768,))
        predictions = keras.layers.Dense(1, activation="sigmoid", name="W1")(inputs)
        model = tf.keras.Model(inputs=inputs, outputs=predictions)
          
        model.compile(optimizer=tf.keras.optimizers.Adam(),
                        loss=tf.keras.losses.binary_crossentropy,
                        metrics=['accuracy', "mae", "mse", tf.keras.metrics.binary_crossentropy
                        , tf.keras.metrics.kld,])
        self._model = model
        self._model.summary()
        self._model_json_string = model.to_json()

    
    def load_h5model(self, h5_model_path):
        self._model.load_weights(h5_model_path)

    def train(self, train_dataset, checkpoint_path="./model/", epochs=1, steps_per_epoch=256):
        # Create checkpoint callback
        cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, 
                                                         save_weights_only=True,
                                                         verbose=1)
        self._model.fit(train_dataset, epochs=epochs, steps_per_epoch=steps_per_epoch,
                  #validation_data=train_dataset, validation_steps=steps_per_epoch*10,
                  callbacks = [cp_callback])  # pass callback to training

    def save_model(self, keras_model_path):
        tf.keras.models.save_model(
            self._model,
            keras_model_path,
            overwrite=True,
            include_optimizer=True
        )
        print("model saved")

    def get_weights(self, fp=None):
        for layer in self._model.layers:
            if not fp:
                print(layer.get_config(), layer.get_weights())
            else:
                name = layer.get_config()['name']
                if name in ['W1']:
                    print(name, file=fp)
                    for weight in layer.get_weights():
                        for w in weight:
                            if isinstance(w, list) or isinstance(w, np.ndarray):
                                print(" ".join([str(wi) for wi in w]), file=fp)
                            else:
                                print(w, file=fp)

    def predict(self, batch):
        if isinstance(batch, list):
            batch = np.array(batch)
        predict = self._model.predict_on_batch(batch)
        return [x[0] for x in predict]

    def file_predict(self, fp, batch_size=128, out_file=None):
        def extract_line(line):
            line_arr = line.strip().split("\t")
            pid = line_arr[0]
            weight = line_arr[1]
            label = line_arr[2]
            feat = [float(x) for x in line_arr[3].strip().split()]
            return (pid + "\t" + weight + "\t" + label, feat)
        if fp:
            if not out_file:
                out_file = sys.stdout
            batch_feat = []
            batch_key = []
            line = fp.readline()
            while line:
                try:
                    key, feat = extract_line(line)
                    if len(batch_feat) >= batch_size:
                        predict = self.predict(batch_feat)
                        if len(predict) == len(batch_key):
                            for i in range(len(predict)):
                                out_str = "{0}\t{1}".format(batch_key[i], str(round(predict[i], 5)))
                                print(out_str, file=out_file)
                        batch_feat = []
                        batch_key = []
                    else:
                        batch_key.append(key)
                        batch_feat.append(feat)
                    line = fp.readline()
                except:
                    line = fp.readline()
                    err = traceback.format_exc()
                    print(err, file=sys.stderr)
                    pass 
            if len(batch_feat) > 0:
                predict = self.predict(batch_feat)
                if len(predict) == len(batch_key):
                    for i in range(len(predict)):
                        out_str = "{0}\t{1}".format(batch_key[i], predict[i])
                        print(out_str, file=out_file)  
                batch_feat = []
                batch_key = []


def train():
    train_dataset = gen_dataset(filenames, shuffle=True, batch_size=1280, repeat_num=150)
    product_model = ScoreModel()
    # 1280 * 256 * 10
    product_model.train(train_dataset, checkpoint_path, epochs=100, steps_per_epoch=256)
    product_model.save_model(keras_model_path)
    for filename in predict_filenames:
        with open(filename) as fp, open(filename + ".out", "w") as wfp:
            product_model.file_predict(fp, out_file=wfp)

def load():
    product_model = ScoreModel()
    product_model.load_h5model(keras_model_path)
    with open('./model/model.txt', "w") as fp:
        product_model.get_weights(fp)

if __name__ == '__main__':
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)

    load()





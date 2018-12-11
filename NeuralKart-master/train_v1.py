import glob
import os
import hashlib
import argparse
from mkdir_p import mkdir_p

from PIL import Image

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv3D
from keras.layers.normalization import BatchNormalization
from keras import optimizers
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping

import matplotlib.pyplot as plt

TRACK_CODES = set(map(lambda s: s.lower(),
    ["ALL", "MR","CM","BC","BB","YV","FS","KTB","RRy","LR","MMF","TT","KD","SL","RRd","WS",
     "BF","SS","DD","DK","BD","TC"]))

def is_valid_track_code(value):
    value = value.lower()
    if value not in TRACK_CODES:
        raise argparse.ArgumentTypeError("%s is an invalid track code" % value)
    return value

OUT_SHAPE = 1

IN_DEPTH = 4
INPUT_WIDTH = 150 #200
INPUT_HEIGHT = 50 #66
INPUT_CHANNELS = 3

VALIDATION_SPLIT = 0.1
USE_REVERSE_IMAGES = False

def customized_loss(y_true, y_pred, loss='euclidean'):
    # Simply a mean squared error that penalizes large joystick summed values
    if loss == 'L2':
        L2_norm_cost = 0.001
        val = K.mean(K.square((y_pred - y_true)), axis=-1) \
            + K.sum(K.square(y_pred), axis=-1) / 2 * L2_norm_cost
    # euclidean distance loss
    elif loss == 'euclidean':
        val = K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))
    return val

def process_observation(frames, steerings, attention=[0.1, 0.1, 0.2, 0.6]):
    # frames = (frames) / 128.0  # normalize from -1. to 1. [-128, 127]
    #x = np.expand_dims(frames, axis=1)
    x = []
    y = []
    for i in range(len(frames)-3):
        x.append(frames[i:i+4])
        y.append(np.sum(steerings[i:i+4]*np.array(attention)))
    return  np.asarray(x), np.array(y)

def create_model(keep_prob=0.6):
    model = Sequential()

    # NVIDIA's model
    model.add(BatchNormalization(input_shape=(IN_DEPTH, INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS)))
    model.add(Conv3D(24, kernel_size=(2, 5, 5), strides=(2, 2, 2), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv3D(36, kernel_size=(2, 5, 5), strides=(1, 2, 2), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv3D(48, kernel_size=(1, 5, 5), strides=(1, 2, 2), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv3D(64, kernel_size=(1, 3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv3D(64, kernel_size=(1, 3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(1164, activation='relu'))
    drop_out = 1 - keep_prob
    model.add(Dropout(drop_out))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(drop_out))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(drop_out))
    model.add(Dense(10, activation='relu'))
    model.add(Dropout(drop_out))
    model.add(Dense(OUT_SHAPE, activation='softsign', name="predictions"))

    return model

def is_validation_set(string):
    string_hash = hashlib.md5(string.encode('utf-8')).digest()
    return int.from_bytes(string_hash[:2], byteorder='big') / 2**16 > VALIDATION_SPLIT

def load_training_data(track):
    X_train, y_train = [], []
    X_val, y_val = [], []

    if track == 'all':
        recordings = glob.iglob("recordings/*/*/*")
    else:
        recordings = glob.iglob("recordings/{}/*/*".format(track))
        #recordings = ['recordings/KD/GP/play-and-search-7cc2e0d2-3677-4fd8-cf08-e2abf0e74704', 'recordings/MMF/TT/play-and-search-9c264448-fba5-4649-c6b1-1142ec23a4de']

    for recording in recordings:
        filenames = list(glob.iglob('{}/*.png'.format(recording)))
        filenames.sort(key=lambda f: int(os.path.basename(f)[:-4]))

        steering = [float(line) for line in open(
            ("{}/steering.txt").format(recording)).read().splitlines()]

        assert len(filenames) == len(steering), "For recording %s, the number of steering values does not match the number of images." % recording

        for file, steer in zip(filenames, steering):
            assert steer >= -1 and steer <= 1

            valid = is_validation_set(file)
            valid_reversed = is_validation_set(file + '_flipped')

            im = Image.open(file).resize((INPUT_WIDTH, INPUT_HEIGHT))
            im_arr = np.frombuffer(im.tobytes(), dtype=np.uint8)
            im_arr = im_arr.reshape((INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS))

            if valid:
                X_train.append(im_arr)
                y_train.append(steer)
            else:
                X_val.append(im_arr)
                y_val.append(steer)

            if USE_REVERSE_IMAGES:
                im_reverse = im.transpose(Image.FLIP_LEFT_RIGHT)
                im_reverse_arr = np.frombuffer(im_reverse.tobytes(), dtype=np.uint8)
                im_reverse_arr = im_reverse_arr.reshape((INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS))

                if valid_reversed:
                    X_train.append(im_reverse_arr)
                    y_train.append(-steer)
                else:
                    X_val.append(im_reverse_arr)
                    y_val.append(-steer)

    X_train, y_train = process_observation(X_train, y_train)
    X_val,y_val = process_observation(X_val, y_val)

    #y_train = y_train[3:]
    #y_val = y_val[3:]

    assert len(X_train) == len(y_train)
    assert len(X_val) == len(y_val)

    return np.asarray(X_train), \
        np.asarray(y_train).reshape((len(y_train), 1)), \
        np.asarray(X_val), \
        np.asarray(y_val).reshape((len(y_val), 1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('track', type=is_valid_track_code)
    parser.add_argument('-c', '--cpu', action='store_true', help='Force Tensorflow to use the CPU.', default=False)
    args = parser.parse_args()

    if args.cpu:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # Load Training Data
    X_train, y_train, X_val, y_val = load_training_data(args.track)

    print(X_train.shape[0], 'training samples.')
    print(X_val.shape[0], 'validation samples.')

    # Training loop variables
    epochs = 5 #100
    batch_size = 10 #50

    model = create_model()

    mkdir_p("new_weights")
    weights_file = "new_weights/{}.hdf5".format(args.track)
    if os.path.isfile(weights_file):
        model.load_weights(weights_file)

    model.compile(loss=customized_loss, optimizer=optimizers.adam(lr=0.0001))
    checkpointer = ModelCheckpoint(
        monitor='val_loss', filepath=weights_file, verbose=1, save_best_only=True, mode='min')
    earlystopping = EarlyStopping(monitor='val_loss', patience=20)

    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,
              shuffle=True, validation_data=(X_val, y_val), callbacks=[checkpointer, earlystopping])

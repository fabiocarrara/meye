# -*- coding: utf-8 -*-
""" MEye: Semantic Segmentation """

import argparse
import os

import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

from dataloader import DataGen, visualize
from model import build_model


def main(args):
    data = os.path.join(args.data, 'annotations.csv')
    data = pd.read_csv(data)

    # TRAIN/VAL/TEST: 70-20-10 %
    train_data, valtest_data = train_test_split(data, test_size=.3, shuffle=True)
    val_data, test_data = train_test_split(valtest_data, test_size=.33)
    lengths = map(len, (data, train_data, val_data, test_data))
    print("Total: {} - Train / Val / Test: {} / {} / {}".format(*lengths))

    x_shape = (args.resolution, args.resolution, 1)
    y_shape = (args.resolution, args.resolution, 2)

    train_gen = DataGen(train_data, args.data, x_shape, args.batch_size)
    val_gen = DataGen(val_data, args.data, x_shape, args.batch_size, deterministic=True)
    test_gen = DataGen(test_data, args.data, x_shape, args.batch_size, deterministic=True)

    x, y = train_gen[0]
    # visualize(x, y)

    model = build_model(x_shape, y_shape)
    model.summary()

    model.compile(optimizer='adam', loss='binary_crossentropy')

    # the checkpointer automatically saves the model which gave the best metric value on the validation set
    checkpointer = ModelCheckpoint('best_weights.hdf5', monitor='val_loss', save_best_only=True, save_weights_only=True)

    # the lr_scheduler automatically reduces the learning rate when reaching a plateau in the validation loss
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', patience=20)

    model.fit_generator(train_gen,
                        steps_per_epoch=len(train_gen),
                        epochs=50,
                        callbacks=[checkpointer, lr_scheduler],
                        validation_data=val_gen,
                        validation_steps=len(val_gen),
                        workers=4)

    model.load_weights('best_weights.hdf5')
    best_val_loss = model.evaluate_generator(test_gen, len(test_gen))
    print('Best Test Loss: {:.4g}'.format(best_val_loss))

    # let's show the worst batch of test (the one with max loss)
    # i = np.array([model.test_on_batch(*i) for i in test_gen]).argmax()

    # x, y = test_gen[i]
    # p = model.predict(x)  # > 0.5
    # visualize(x, p)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-d', '--data', default='data/NN_fullframe', help='Data directory')
    parser.add_argument('-r', '--resolution', type=int, default=128, help='Input image resolution')
    parser.add_argument('-b', '--batch-size', type=int, default=8, help='Batch size')

    args = parser.parse_args()
    main(args)

# -*- coding: utf-8 -*-
""" MEye: Semantic Segmentation """

import argparse
import os

os.sys.path += ['expman']
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

import math
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, CSVLogger
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_curve, auc, precision_recall_curve, average_precision_score
from adabelief_tf import AdaBeliefOptimizer

from dataloader import get_loader, load_datasets
from model import build_model
from utils import visualize
from expman import Experiment


def iou_coef(y_true, y_pred, smooth=0.001, thr=0.5):
    y_pred = K.cast(y_pred > thr, 'float32')
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1, 2, 3])
    union = K.sum(y_true, [1, 2, 3]) + K.sum(y_pred, [1, 2, 3]) - intersection
    iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
    return iou

def dice_coef(y_true, y_pred, smooth=0.001, thr=0.5):
    y_pred = K.cast(y_pred > thr, 'float32')
    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
    dice = K.mean((2. * intersection + smooth) / (union + smooth), axis=0)
    return dice


def main(args):
    exp = Experiment(args, ignore=('eval_only', 'epochs', 'resume'))
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    data = load_datasets(args.data)

    # TRAIN/VAL/TEST SPLIT
    val_subjects = (6, 9, 11, 13, 16, 28, 30, 48, 49)
    test_subjects = (3, 4, 19, 38, 45, 46, 51, 52)
    train_data = data[~data['sub'].isin(val_subjects + test_subjects)]
    val_data = data[data['sub'].isin(val_subjects)]
    test_data = data[data['sub'].isin(test_subjects)]

    lengths = map(len, (data, train_data, val_data, test_data))
    print("Total: {} - Train / Val / Test: {} / {} / {}".format(*lengths))

    x_shape = (args.resolution, args.resolution, 1)
    y_shape = (args.resolution, args.resolution, 1)

    train_gen, _ = get_loader(train_data, batch_size=args.batch_size, shuffle=True, augment=True, x_shape=x_shape)
    val_gen, val_categories = get_loader(val_data, batch_size=args.batch_size, x_shape=x_shape)
    test_gen, test_categories = get_loader(test_data, batch_size=1, x_shape=x_shape)

    # x, y = train_gen[0]
    # visualize(x, y)
    config = vars(args)
    model = build_model(x_shape, y_shape, config)
    model.summary()

    optimizer = AdaBeliefOptimizer(learning_rate=args.lr)
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics={'mask': [iou_coef, dice_coef],
                           'tags': 'binary_accuracy'})

    log = exp.path_to('log.csv')
    best_ckpt_path = exp.path_to('best_weights.h5')
    last_ckpt_path = exp.path_to('last_weights.h5')

    # the checkpointer automatically saves the model which gave the best metric value on the validation set
    best_checkpointer = ModelCheckpoint(best_ckpt_path, monitor='val_loss', save_best_only=True, save_weights_only=True)
    last_checkpointer = ModelCheckpoint(last_ckpt_path, save_best_only=False, save_weights_only=False)
    logger = CSVLogger(log, append=args.resume)

    callbacks = [best_checkpointer, last_checkpointer, logger]

    initial_epoch = 0
    if args.resume and os.path.exists(last_ckpt_path):
        custom_objects={'AdaBeliefOptimizer': AdaBeliefOptimizer, 'iou_coef': iou_coef, 'dice_coef': dice_coef}
        model = tf.keras.models.load_model(last_ckpt_path, custom_objects=custom_objects)
        initial_epoch = len(pd.read_csv(log))

    if not args.eval_only:
        model.fit(train_gen,
                  epochs=args.epochs,
                  callbacks=callbacks,
                  initial_epoch=initial_epoch,
                  steps_per_epoch=len(train_gen),
                  validation_data=val_gen,
                  validation_steps=len(val_gen))

    # load best checkpoint
    model.load_weights(best_ckpt_path)

    # today = datetime.datetime.now().strftime('%Y-%m-%d')
    best_model_path = 'meye-segmentation_' \
                      'i{resolution}_' \
                      's{num_stages}_' \
                      'c{num_conv}_' \
                      'f{num_filters}_' \
                      'g{grow_factor}_' \
                      'a-{up_activation}' \
                      't-{conv_type}' \
                      'p-{use_aspp}'  \
                      '/'.format_map(vars(args))

    best_model_path = exp.path_to(best_model_path)
    if not os.path.exists(best_model_path):
        tf.keras.models.save_model(model, best_model_path, include_optimizer=False)

    # evaluation on test set
    evaluate.evaluate(exp, force=make_eval)
    exit()


if __name__ == '__main__':
    # default_data = ['data/2p-dataset', 'data/H-dataset', 'data/NN_fullframe_extended', 'data/NN_mixed_dataset']
    default_data = ['data/NN_mixed_dataset_new']

    parser = argparse.ArgumentParser(description='')
    # data params
    parser.add_argument('-d', '--data', nargs='+', default=default_data, help='Data directory (may be multiple)')
    parser.add_argument('-r', '--resolution', type=int, default=128, help='Input image resolution')

    # model params
    parser.add_argument('--num-stages', type=int, default=4, help='number of down-up sample stages')
    parser.add_argument('--num-conv', type=int, default=1, help='number of convolutions per stage')
    parser.add_argument('--num-filters', type=int, default=16, help='number of conv filter at first stage')
    parser.add_argument('--grow-factor', type=float, default=1.0,
                        help='# filters at stage i = num-filters * grow-factor ** i')
    parser.add_argument('--up-activation', default='relu', choices=('relu', 'lrelu'),
                        help='activation in upsample stages')
    parser.add_argument('--conv-type', default='conv', choices=('conv', 'bn-conv', 'sep-conv', 'sep-bn-conv'),
                        help='convolution type')
    parser.add_argument('--use-aspp', default=False, action='store_true', help='Use Atrous Spatial Pyramid Pooling')

    # train params
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('-b', '--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('-e', '--epochs', type=int, default=750, help='Number of training epochs')
    parser.add_argument('-s', '--seed', type=int, default=23, help='Random seed')
    parser.add_argument('--resume', default=False, action='store_true', help='Resume training')

    # other
    parser.add_argument('--eval-only', default=False, action='store_true', help='Evaluate only')

    args = parser.parse_args()
    main(args)

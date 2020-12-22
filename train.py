# -*- coding: utf-8 -*-
""" MEye: Semantic Segmentation """

import argparse
import os

os.sys.path += ['expman']
import matplotlib

matplotlib.use('Agg')

import numpy as np
import pandas as pd
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from dataloader import DataGen, load_datasets
from model import build_model
from utils import visualize
from expman import Experiment


def main(args):
    exp = Experiment(args, ignore=('eval_only', 'epochs', 'resume'))
    np.random.seed(args.seed)

    data = load_datasets(args.data)

    # TRAIN/VAL/TEST: 70-20-10 %
    train_data, valtest_data = train_test_split(data, test_size=.3, shuffle=True)
    val_data, test_data = train_test_split(valtest_data, test_size=.33)
    lengths = map(len, (data, train_data, val_data, test_data))
    print("Total: {} - Train / Val / Test: {} / {} / {}".format(*lengths))

    x_shape = (args.resolution, args.resolution, 1)
    y_shape = (args.resolution, args.resolution, 2)

    train_gen = DataGen(train_data, x_shape=x_shape, batch_size=args.batch_size)
    val_gen = DataGen(val_data, x_shape=x_shape, batch_size=args.batch_size, deterministic=True, no_pad=True)
    test_gen = DataGen(test_data, x_shape=x_shape, batch_size=args.batch_size, deterministic=True, no_pad=True)

    # x, y = train_gen[0]
    # visualize(x, y)
    config = vars(args)
    model = build_model(x_shape, y_shape, config)
    model.summary()

    model.compile(optimizer='sgd', loss='binary_crossentropy', metrics={'out_tags': 'binary_accuracy'})

    log = exp.path_to('log.csv')
    best_ckpt_path = exp.path_to('best_weights.h5')
    last_ckpt_path = exp.path_to('last_weights.h5')

    # today = datetime.datetime.now().strftime('%Y-%m-%d')
    best_model_path = 'meye-segmentation_' \
                      'i{resolution}_' \
                      's{num_stages}_' \
                      'c{num_conv}_' \
                      'f{num_filters}_' \
                      'g{grow_factor}_' \
                      'a-{up_activation}' \
                      '.hdf5'.format_map(vars(args))

    best_model_path = exp.path_to(best_model_path)

    # the checkpointer automatically saves the model which gave the best metric value on the validation set
    best_checkpointer = ModelCheckpoint(best_ckpt_path, monitor='val_loss', save_best_only=True, save_weights_only=True)
    last_checkpointer = ModelCheckpoint(last_ckpt_path, save_best_only=False, save_weights_only=True)
    logger = CSVLogger(log)
    # the lr_scheduler automatically reduces the learning rate when reaching a plateau in the validation loss
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', patience=100)
    callbacks = [best_checkpointer, last_checkpointer, logger, lr_scheduler]

    initial_epoch = 0
    if args.resume:
        model.load_weights(last_ckpt_path)
        initial_epoch = len(pd.read_csv(log))

    if not args.eval_only:
        model.fit_generator(train_gen,
                            steps_per_epoch=len(train_gen),
                            initial_epoch=initial_epoch,
                            epochs=args.epochs,
                            callbacks=callbacks,
                            validation_data=val_gen,
                            validation_steps=len(val_gen),
                            workers=4)

    model.load_weights(best_ckpt_path)
    if not os.path.exists(best_model_path):
        model.save(best_model_path)

    # best_test_loss, best_test_acc = model.evaluate_generator(test_gen, len(test_gen))
    best_metrics = model.evaluate_generator(test_gen, len(test_gen))
    best_test_loss, best_test_acc = best_metrics[:3], best_metrics[3]
    # print('Best Test Loss: {:.4g}'.format(best_test_loss))
    print('Best Test Loss:', best_test_loss)
    print('Best Test Acc:', best_test_acc)
    
    best_pred, targets = zip(*[(model.predict_on_batch(x)[1], y[1]) for x, y in test_gen])
    best_pred = np.concatenate(best_pred) > .5
    targets = np.concatenate(targets).astype(np.float32)
    
    print(classification_report(targets, best_pred, target_names=['eye', 'blink']))

    # let's show the worst batch of test (the one with max loss)
    loss_per_batch = np.array([model.test_on_batch(*i)[:3] for i in test_gen]).sum(1)
    best = loss_per_batch.argmin()
    worst = loss_per_batch.argmax()

    x, y = test_gen[worst]
    p = model.predict(x)  # > 0.5
    viz_path = exp.path_to('worst_test_batch.png')
    visualize(x, p, out=viz_path)

    x, y = test_gen[best]
    p = model.predict(x)  # > 0.5
    viz_path = exp.path_to('best_test_batch.png')
    visualize(x, p, out=viz_path)


if __name__ == '__main__':
    # default_data = ['data/2p-dataset', 'data/H-dataset', 'data/NN_fullframe_extended', 'data/NN_mixed_dataset']
    default_data = ['data/NN_mixed_dataset_new']

    parser = argparse.ArgumentParser(description='')
    # data params
    parser.add_argument('-d', '--data', nargs='+', default=default_data, help='Data directory (may be multiple)')
    parser.add_argument('-r', '--resolution', type=int, default=128, help='Input image resolution')

    # model params
    parser.add_argument('--num-stages', type=int, default=2, help='number of down-up sample stages')
    parser.add_argument('--num-conv', type=int, default=1, help='number of convolutions per stage')
    parser.add_argument('--num-filters', type=int, default=16, help='number of conv filter at first stage')
    parser.add_argument('--grow-factor', type=int, default=1,
                        help='# filters at stage i = num-filters * grow-factor ** i')
    parser.add_argument('--up-activation', default='relu', choices=('relu', 'lrelu'),
                        help='activation in upsample stages')

    # train params
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('-b', '--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('-e', '--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('-s', '--seed', type=int, default=23, help='Random seed')
    parser.add_argument('--resume', default=False, action='store_true', help='Resume training')

    # other
    parser.add_argument('--eval-only', default=False, action='store_true', help='Evaluate only')

    args = parser.parse_args()
    main(args)

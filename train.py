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

    train_gen, _ = get_loader(train_data, x_shape=x_shape, batch_size=args.batch_size, shuffle=True, sample_weights=False)
    val_gen, val_categories = get_loader(val_data, x_shape=x_shape, batch_size=args.batch_size, deterministic=True)
    test_gen, test_categories = get_loader(test_data, x_shape=x_shape, batch_size=1, deterministic=True)

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
                      '/'.format_map(vars(args))

    best_model_path = exp.path_to(best_model_path)
    if not os.path.exists(best_model_path):
        tf.keras.models.save_model(model, best_model_path, include_optimizer=False)

    # evaluation on test set
    prediction_dir = exp.path_to('test_pred')
    os.makedirs(prediction_dir, exist_ok=True)

    x_masks = []
    y_masks, y_tags = [], []
    pred_masks, pred_tags = [], []
    loss_per_sample = []

    for x, y in test_gen:
        sample_loss = model.test_on_batch(x, reset_metrics=True)
        loss_per_sample.append(sample_loss)

        p_mask, p_tags = model.predict_on_batch(x)
        pred_masks.append(p_mask)
        pred_tags.append(p_tags)
        y_masks.append(y['mask'].numpy())
        y_tags.append(y['tags'].numpy())
        x_masks.append(x.numpy())

    loss_per_sample = np.array(loss_per_sample)
    pred_masks = np.concatenate(pred_masks)
    pred_tags = np.concatenate(pred_tags)
    y_masks = np.concatenate(y_masks)
    y_tags = np.concatenate(y_tags)
    x_masks = np.concatenate(x_masks)

    thrs = np.linspace(0, 1, 101)
    ious = [iou_coef(y_masks, pred_masks, thr=thr).numpy() for thr in thrs]
    dices = [dice_coef(y_masks, pred_masks, thr=thr).numpy() for thr in thrs]

    mask_metrics = pd.DataFrame({'iou': ious, 'dice': dices, 'thr': thrs})
    print(mask_metrics.max(axis=0))
    mask_metrics.to_csv(exp.path_to('test_pred/mask_metrics.csv'))

    def _filter_by_closeness(a, eps=10e-3):
        keep = []
        prev = np.array([-1, -1])
        for row in a.drop('thr', axis=1).values:
            if (np.abs(prev - row) > eps).any():
                keep.append(True)
                prev = row
            else:
                keep.append(False)
        return a[keep]

    def _weighted_roc_pr(y_true, y_scores, label, outdir, simplify=False):
        npos = y_true.sum()
        nneg = len(y_true) - npos
        pos_weight = nneg / npos
        print(label, 'Tot:', len(y_true), 'P:', npos, 'N:', nneg, 'N/P:', pos_weight)
        sample_weight = np.where(y_true, pos_weight, 1)

        fpr, tpr, thr = roc_curve(y_true, y_scores, sample_weight=sample_weight)
        auc_score = auc(fpr, tpr)
        print(label, 'AuROC:', auc_score)

        roc_metrics = pd.Series({'npos': npos, 'nneg': nneg, 'nneg_over_npos': pos_weight, 'roc_auc': auc_score})
        roc_metrics_file = os.path.join(outdir, '{}_roc_metrics.csv'.format(label))
        roc_metrics.to_csv(roc_metrics_file, index=False)

        roc = pd.DataFrame({'fpr': fpr, 'tpr': tpr, 'thr': thr})
        if simplify:
            full_roc_file = os.path.join(outdir, '{}_roc_curve_full.csv.gz'.format(label))
            roc.to_csv(full_roc_file, index=False)
            roc = _filter_by_closeness(roc)

        roc_file = os.path.join(outdir, '{}_roc_curve.csv'.format(label))
        roc.to_csv(roc_file, index=False)

        roc.plot(x='fpr', y='tpr', xlim=(0,1), ylim=(0,1))
        roc_plot_file = os.path.join(outdir, '{}_roc.pdf'.format(label))
        plt.savefig(roc_plot_file)
        plt.close()

        precision, recall, thr = precision_recall_curve(y_true, y_scores, sample_weight=sample_weight)
        f1_score = 2 * precision * recall / (precision + recall)
        pr_auc = auc(recall, precision)

        pr_metrics = pd.Series({'npos': npos, 'nneg': nneg, 'nneg_over_npos': pos_weight, 'pr_auc': pr_auc})
        pr_metrics_file = os.path.join(outdir, '{}_pr_metrics.csv'.format(label))
        pr_metrics.to_csv(pr_metrics_file, index=False)

        thr = np.append(thr, [thr[-1]])
        pr = pd.DataFrame({'precision': precision, 'recall': recall, 'f1_score': f1_score, 'thr': thr})
        if simplify:
            full_pr_file = os.path.join(outdir, '{}_pr_curve_full.csv.gz'.format(label))
            pr.to_csv(full_pr_file, index=False)
            pr = _filter_by_closeness(pr)

        pr_file = os.path.join(outdir, '{}_pr_curve.csv'.format(label))
        pr.to_csv(pr_file, index=False)

        pr.plot(x='recall', y='precision', xlim=(0, 1), ylim=(0, 1))
        pr_plot_file = os.path.join(outdir, '{}_pr.pdf'.format(label))
        plt.savefig(pr_plot_file)
        plt.close()

        print(label, 'AuPR:', pr_auc, 'AvgP:', average_precision_score(y_true, y_scores, sample_weight=sample_weight))

    _weighted_roc_pr(y_masks.ravel(), pred_masks.ravel(), 'all_pupil', prediction_dir, simplify=True)
    _weighted_roc_pr(y_tags[:, 0], pred_tags[:, 0], 'all_eye', prediction_dir)
    _weighted_roc_pr(y_tags[:, 1], pred_tags[:, 1], 'all_blink', prediction_dir)

    def _save_best_worst_samples(x_masks, ys, ps, losses, label, outdir, k=5):
        rank = losses.argsort()
        topk, bottomk = rank[:k], rank[-k:]

        y_masks, y_tags = ys
        pred_masks, pred_tags = ps

        for i, (x, ym, yt, pm, pt) in enumerate(
                zip(x_masks[topk], y_masks[topk], y_tags[topk], pred_masks[topk], pred_tags[topk])):
            combined_m = np.concatenate((pm, ym), axis=-1)
            combined_t = np.concatenate((pt, yt), axis=-1)
            combined_y = (combined_m[None, ...], combined_t[None, ...])
            out = os.path.join(outdir, '{}_top{}_sample.png'.format(label, i))
            visualize(x[None, ...], combined_y, out=out, thr=0.5, n_cols=1, width=5)

        for i, (x, ym, yt, pm, pt) in enumerate(
                zip(x_masks[bottomk], y_masks[bottomk], y_tags[bottomk], pred_masks[bottomk], pred_tags[bottomk])):
            combined_m = np.concatenate((pm, ym), axis=-1)
            combined_t = np.concatenate((pt, yt), axis=-1)
            combined_y = (combined_m[None, ...], combined_t[None, ...])
            out = os.path.join(outdir, '{}_bottom{}_sample.png'.format(label, i))
            visualize(x[None, ...], combined_y, out=out, thr=0.5, n_cols=1, width=5)

    for cat in np.unique(test_categories):
        cat_outdir = os.path.join(prediction_dir, cat)
        os.makedirs(cat_outdir, exist_ok=True)

        selector = test_categories == cat
        _weighted_roc_pr(y_masks[selector].ravel(), pred_masks[selector].ravel(), '{}_pupil'.format(cat), cat_outdir, simplify=True)
        _weighted_roc_pr(y_tags[selector, 0], pred_tags[selector, 0], '{}_eye'.format(cat), cat_outdir)
        _weighted_roc_pr(y_tags[selector, 1], pred_tags[selector, 1], '{}_blink'.format(cat), cat_outdir)

        cat_x = x_masks[selector]
        cat_y = (y_masks[selector], y_tags[selector])
        cat_p = (pred_masks[selector], pred_tags[selector])
        cat_losses = loss_per_sample[selector, 1] # .sum(1)
        _save_best_worst_samples(cat_x, cat_y, cat_p, cat_losses, cat, cat_outdir, k=5)


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
    parser.add_argument('--grow-factor', type=float, default=1.0,
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

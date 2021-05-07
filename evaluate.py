# -*- coding: utf-8 -*-
""" MEye: Semantic Segmentation """

import argparse
import os
os.sys.path += ['expman', 'models/deeplab']
import expman

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_curve, auc, precision_recall_curve, average_precision_score
from adabelief_tf import AdaBeliefOptimizer
from glob import glob
from tqdm import tqdm
from PIL import Image

from deeplabv3p.models.deeplabv3p_mobilenetv3 import hard_swish
from dataloader import get_loader, load_datasets
from utils import visualize, visualizable


def iou_coef(y_true, y_pred, smooth=0.001, thr=None):
    y_pred = K.cast(y_pred > thr, 'float32') if thr is not None else y_pred
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1, 2, 3])
    union = K.sum(y_true, [1, 2, 3]) + K.sum(y_pred, [1, 2, 3]) - intersection
    iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
    return iou


def dice_coef(y_true, y_pred, smooth=0.001, thr=None):
    y_pred = K.cast(y_pred > thr, 'float32') if thr is not None else y_pred
    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
    dice = K.mean((2. * intersection + smooth) / (union + smooth), axis=0)
    return dice


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

    roc.plot(x='fpr', y='tpr', xlim=(0, 1), ylim=(0, 1))
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


# https://github.com/tensorflow/tensorflow/issues/32809#issuecomment-768977280
from tensorflow.python.framework.convert_to_constants import  convert_variables_to_constants_v2_as_graph
def get_flops(model):
    concrete = tf.function(lambda inputs: model(inputs))
    concrete_func = concrete.get_concrete_function(
        [tf.TensorSpec([1, *inputs.shape[1:]]) for inputs in model.inputs])
    frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(concrete_func)
    with tf.Graph().as_default() as graph:
        tf.graph_util.import_graph_def(graph_def, name='')
        run_meta = tf.compat.v1.RunMetadata()
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd="op", options=opts)

    tf.compat.v1.reset_default_graph()
    return flops.total_float_ops


def evaluate(exp, force=False):

    ckpt_path = exp.path_to('best_model.h5')

    custom_objects = {'AdaBeliefOptimizer': AdaBeliefOptimizer, 'iou_coef': iou_coef, 'dice_coef': dice_coef, 'hard_swish': hard_swish}
    model = tf.keras.models.load_model(ckpt_path, custom_objects=custom_objects)

    # get flops
    flop_params_path = exp.path_to('flops_nparams.csv')
    if force or not os.path.exists(flop_params_path):
        model.compile()
        tf.keras.models.save_model(model, 'tmp_model', overwrite=True, include_optimizer=False)
        stripped_model = tf.keras.models.load_model('tmp_model')
        flops = get_flops(stripped_model)
        nparams = stripped_model.count_params()
        del stripped_model
        print('FLOPS:', flops)
        print('#PARAMS:', nparams)
        pd.DataFrame({'flops': flops, 'nparams': nparams}, index=[0]).to_csv(flop_params_path)

    model.compile(loss='binary_crossentropy', metrics={'mask': [iou_coef, dice_coef], 'tags': 'binary_accuracy'})

    params = exp.params
    np.random.seed(params.seed)
    tf.random.set_seed(params.seed)

    data = load_datasets(params.data)

    # TRAIN/VAL/TEST SPLIT
    if params.split == 'subjects':  # by SUBJECTS
        # val_subjects = (6, 9, 11, 13, 16, 28, 30, 48, 49)
        test_subjects = (3, 4, 19, 38, 45, 46, 51, 52)
        # train_data = data[~data['sub'].isin(val_subjects + test_subjects)]
        # val_data = data[data['sub'].isin(val_subjects)]
        test_data = data[data['sub'].isin(test_subjects)]

    elif params.split == 'random':  # 70-20-10 %
        _, valtest_data = train_test_split(data, test_size=.3, shuffle=True)
        _, test_data = train_test_split(valtest_data, test_size=.33)

    x_shape = (params.resolution, params.resolution, 1)
    test_gen, test_categories = get_loader(test_data, batch_size=1, x_shape=x_shape)

    prediction_dir = exp.path_to('test_pred')
    os.makedirs(prediction_dir, exist_ok=True)

    loss_per_sample = None

    def _get_test_predictions(test_gen, model):
        x_masks = []
        y_masks, y_tags = [], []
        pred_masks, pred_tags = [], []
        loss_per_sample = []

        for x, y in tqdm(test_gen, desc='TEST'):
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

        return loss_per_sample, x_masks, y_masks, y_tags, pred_masks, pred_tags


    mask_metrics_path = exp.path_to('test_pred/mask_metrics.csv')
    if force or not os.path.exists(mask_metrics_path):
        if loss_per_sample is None:
            loss_per_sample, x_masks, y_masks, y_tags, pred_masks, pred_tags = _get_test_predictions(test_gen, model)

        thrs = np.linspace(0, 1, 101)
        ious = [iou_coef(y_masks, pred_masks, thr=thr).numpy() for thr in thrs]
        dices = [dice_coef(y_masks, pred_masks, thr=thr).numpy() for thr in thrs]

        best_thr = max(zip(dices, thrs))[1]

        mask_metrics = pd.DataFrame({'iou': ious, 'dice': dices, 'thr': thrs})
        print(mask_metrics.max(axis=0))
        mask_metrics.to_csv(mask_metrics_path)
    else:
        mask_metrics = pd.read_csv(mask_metrics_path, index_col=0)
        best_thr = mask_metrics.loc[mask_metrics.dice.idxmax(), 'thr']

    if force:
        if loss_per_sample is None:
            loss_per_sample, x_masks, y_masks, y_tags, pred_masks, pred_tags = _get_test_predictions(test_gen, model)
        # _weighted_roc_pr(y_masks.ravel(), pred_masks.ravel(), 'all_pupil', prediction_dir, simplify=True)
        _weighted_roc_pr(y_tags[:, 0], pred_tags[:, 0], 'all_eye', prediction_dir)
        _weighted_roc_pr(y_tags[:, 1], pred_tags[:, 1], 'all_blink', prediction_dir)

    filenames = ('top_samples.png', 'bottom_samples.png', 'random_samples.png')
    if force or any(not os.path.exists(os.path.join(prediction_dir, f)) for f in filenames):
        if loss_per_sample is None:
            loss_per_sample, x_masks, y_masks, y_tags, pred_masks, pred_tags = _get_test_predictions(test_gen, model)

        k = 5
        best_selector = []
        worst_selector = []
        random_selector = []

        idx = np.arange(len(test_data))
        for cat in np.unique(test_categories):
            cat_outdir = os.path.join(prediction_dir, cat)
            os.makedirs(cat_outdir, exist_ok=True)

            selector = test_categories == cat
            # _weighted_roc_pr(y_masks[selector].ravel(), pred_masks[selector].ravel(), '{}_pupil'.format(cat), cat_outdir, simplify=True)
            _weighted_roc_pr(y_tags[selector, 0], pred_tags[selector, 0], '{}_eye'.format(cat), cat_outdir)
            _weighted_roc_pr(y_tags[selector, 1], pred_tags[selector, 1], '{}_blink'.format(cat), cat_outdir)

            cat_losses = loss_per_sample[selector, 1]
            rank = cat_losses.argsort()
            topk, bottomk = rank[:k], rank[-k:]

            best_selector += idx[selector][topk].tolist()
            worst_selector += idx[selector][bottomk].tolist()
            random_selector += np.random.choice(idx[selector], k, replace=False).tolist()

        # topk-bottomk images
        selectors = (best_selector, worst_selector, random_selector)
        for selector, outfile in zip(selectors, filenames):
            combined_m = np.concatenate((pred_masks[selector], y_masks[selector]), axis=-1)[:, :, :, ::-1]
            combined_t = np.concatenate((pred_tags[selector], y_tags[selector]), axis=-1)
            combined_y = (combined_m, combined_t)
            out = os.path.join(prediction_dir, outfile)
            visualize(x_masks[selector], combined_y, out=out, thr=best_thr, n_cols=k, width=10)

            for i, (xi, yi_mask) in enumerate(zip(x_masks[selector], combined_m)):
                img = visualizable(xi, yi_mask, thr=best_thr)
                img = (img * 255).astype(np.uint8)
                out = os.path.join(prediction_dir, outfile[:-4])
                os.makedirs(out, exist_ok=True)
                out = os.path.join(out, f'{i}.png')
                Image.fromarray(img).save(out)


def main(args):
    for exp in expman.gather(args.run).filter(args.filter):
        print(exp)
        evaluate(exp, force=args.force)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Run')
    # data params
    parser.add_argument('run', help='Run(s) directory')
    parser.add_argument('-f', '--filter', default={}, type=expman.exp_filter)
    parser.add_argument('--force', default=False, action='store_true', help='Force metrics recomputation')

    args = parser.parse_args()
    main(args)

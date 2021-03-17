import argparse
import math
import os
os.sys.path += ['expman']

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib.image import imread
from matplotlib.backends.backend_pdf import PdfPages
from glob import glob

import expman

def ee_old(args):
    sns.set_theme(context='notebook', style='whitegrid')

    exps = expman.gather(args.run).filter(args.filter)
    mask_metrics = exps.collect('test_pred/mask_metrics.csv').groupby('exp_id')[['dice', 'iou']].max()
    flops_nparams = exps.collect('flops_nparams.csv')

    data = pd.merge(mask_metrics, flops_nparams, on='exp_id')

    data['dice'] *= 100

    named_data = data.rename({
        'nparams': '# Params',
        'dice': 'Dice Coeff. (%)',
        'conv_type': 'Conv. Type ($t$)',
        'grow_factor': r'$\gamma$',
        'num_filters': r'$k$',
        'flops': 'FLOPs',
        'num_stages': r'$s$',
    }, axis=1).replace({
        #'conv': 'Conv.',
        #'bn-conv': 'Conv. + BN',
        #'sep-conv': 'Separable Conv.',
        #'sep-bn-conv': 'Separable Conv. + BN'
        # 'conv': 'Conv.',
        'bn-conv': 'conv-bn',
        'sep-conv': 'sep-conv',
        'sep-bn-conv': 'sep-conv-bn'
    })

    fig, axes = plt.subplots(1, 1, figsize=(5, 3.5))
    # (ax1, ax2) = axes
    ax2 = axes
    # sns.scatterplot(data=named_data, x='# Params', y='Dice Coeff. (%)', hue='Conv. Type', size=r'$\gamma$', style=r'$K$', ax=ax1)
    # sns.lineplot(data=data, x='nparams', y='dice', hue='conv_type', size='grow_factor', markers='num_filters', ax=ax1)
    # ax1.legend(bbox_to_anchor=(1.01, 1), borderaxespad=0)

    #sns.scatterplot(data=named_data, x='FLOPs', y='Dice Coeff. (%)',
    #                hue='Conv. Type', hue_order=['Conv.', 'Conv. + BN', 'Separable Conv.', 'Separable Conv. + BN'],
    #                size=r'$\gamma$', sizes=(40, 140),
    #                style=r'$K$', ax=ax2)
    sns.lineplot(data=named_data,
                 x='FLOPs', y='Dice Coeff. (%)',
                 hue='Conv. Type ($t$)',
                 # hue_order=['Conv.', 'Conv. + BN', 'Separable Conv.', 'Separable Conv. + BN'],
                 hue_order=['conv', 'conv-bn', 'sep-conv', 'sep-conv-bn'],
                 # size=r'$s$',
                 style=r'$k$', markers=True, markersize=9, ax=ax2)
    ax2.legend(ncol=2, loc='best', columnspacing=0) #bbox_to_anchor=(1.01, 1), borderaxespad=0)
    ax2.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    #b, t = ax2.get_ylim()
    #b, t = math.floor(b/10) * 10, math.ceil(t/10) * 10
    ax2.set_ylim(bottom=40)
    # ax2.set_xlim(left=0, right=1.6 * 10**9)
    ax2.set_xscale('symlog')
    ax2.set_xlim(left=0.04 * 10 ** 9, right=2 * 10 ** 9)

    k_formatter = ticker.FuncFormatter(lambda x, pos: '{:g}'.format(x / 10 ** 3) + 'K')
    m_formatter = ticker.FuncFormatter(lambda x, pos: '{:g}'.format(x / 10 ** 6) + 'M')
    b_formatter = ticker.FuncFormatter(lambda x, pos: '{:.2f}'.format(x / 10 ** 9) + 'B')

    # ax2.xaxis.set_major_locator(ticker.SymmetricalLogLocator(base=10, linthresh=2, subs=[2,4,6,8]))
    # ax2.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    # ax2.xaxis.set_minor_locator(ticker.SymmetricalLogLocator(base=10, linthresh=2, subs=[1,1.5,2,3,4,5,6,8]))
    ax2.xaxis.set_minor_locator(ticker.SymmetricalLogLocator(base=10, linthresh=2, subs=[1.5, 2,3,4,5,6,8]))
    ax2.xaxis.set_minor_formatter(b_formatter)
    #xticks = ticker.SymmetricalLogLocator(base=10, linthresh=2, subs=[1, 1.5, 2, 3, 4, 5, 6, 8]).tick_values(0.04 * 10 ** 9, 2 * 10 ** 9)
    #ax2.set_xticks(xticks)
    #ax2.set_xticklabels(['{:.2f}'.format(x / 10 ** 9) + 'B' for x in xticks])
    ax2.grid(which='minor', linestyle='--', color='#eeeeee')

    # ax1.xaxis.set_major_formatter(k_formatter)
    ax2.xaxis.set_major_formatter(b_formatter)
    ax2.tick_params(axis="x", which="both", rotation=90)
    #for text in ax2.get_xminorticklabels():
    #    text.set_fontsize(10)

    plt.savefig(args.output, bbox_inches='tight')

    data['bn'] = data.conv_type.str.contains('bn')
    data['separable'] = data.conv_type.str.contains('sep')
    pivot = pd.pivot_table(data,
                           index=['grow_factor', 'num_filters', 'separable', 'bn'],
                           values=['dice', 'iou', 'nparams', 'flops'])
    pivot = pivot.reindex(columns=['dice', 'iou', 'nparams', 'flops'])
    pivot = pivot.reset_index()

    bool_symb = {False: '', True: r'\cmark'}
    pivot[['separable', 'bn']] = pivot[['separable', 'bn']].replace(bool_symb)

    pivot = pivot.sort_values('flops', ascending=True)

    pivot = pivot.rename(columns={
        'grow_factor': r'$\gamma$',
        'num_filters': r'$K$',
        'separable': 'Sep.',
        'bn': 'BN',
        'dice': 'Dice',
        'iou': 'IoU',
        'nparams': 'Params (K)',
        'flops': 'FLOPs (M)'
    })

    formatters = {
        r'$\gamma$': '{:g}'.format,
        'Dice': '{:.2f}'.format,
        'IoU': '{:.2f}'.format,
        'Params (K)': lambda x: f'{x / 10 ** 3:.1f}',
        'FLOPs (M)': lambda x: f'{x / 10 ** 6:.1f}',
    }

    print(pivot.to_latex(formatters=formatters, index=False, escape=False))


def ee(args):
    sns.set_theme(context='notebook', style='whitegrid')

    exps = expman.gather(args.run).filter(args.filter)
    mask_metrics = exps.collect('test_pred/mask_metrics.csv').groupby('exp_id')[['dice', 'iou']].max()
    flops_nparams = exps.collect('flops_nparams.csv')
    data = pd.merge(mask_metrics, flops_nparams, on='exp_id')
    data['dice'] *= 100

    named_data = data.rename({
        'nparams': '# Params',
        'dice': 'mean Dice Coeff. (%)',
        'conv_type': '$t$ (Conv. Type)',
        'grow_factor': r'$\gamma$',
        'num_filters': '$k$ (# Filters)',
        'flops': 'FLOPs',
        'num_stages': '$s$ (# Stages)',
    }, axis=1).replace({
        #'conv': 'Conv.',
        #'bn-conv': 'Conv. + BN',
        #'sep-conv': 'Separable Conv.',
        #'sep-bn-conv': 'Separable Conv. + BN'
        # 'conv': 'Conv.',
        'bn-conv': 'conv-bn',
        'sep-conv': 'sep-conv',
        'sep-bn-conv': 'sep-conv-bn'
    })

    g = sns.relplot(data=named_data,
                    x='FLOPs', y='mean Dice Coeff. (%)',
                    hue='$t$ (Conv. Type)',
                    # hue_order=['Conv.', 'Conv. + BN', 'Separable Conv.', 'Separable Conv. + BN'],
                    hue_order=['conv', 'conv-bn', 'sep-conv', 'sep-conv-bn'],
                    col='$s$ (# Stages)', style='$k$ (# Filters)', markers=True, markersize=9,
                    kind='line', dashes=True, facet_kws=dict(despine=False, legend_out=False), legend=True,
                    height=3.8, aspect=1.3, markeredgecolor='white')

    k_formatter = ticker.FuncFormatter(lambda x, pos: '{:g}'.format(x / 10 ** 3) + 'K')
    m_formatter = ticker.FuncFormatter(lambda x, pos: '{:g}'.format(x / 10 ** 6) + 'M')
    b_formatter = ticker.FuncFormatter(lambda x, pos: '{:.2f}'.format(x / 10 ** 9) + 'B')

    h, l = g.axes.flatten()[0].get_legend_handles_labels()
    for hi in h:
        hi.set_markeredgecolor('white')
    g.axes.flatten()[0].legend_.remove()
    g.fig.legend(h, l, ncol=2, bbox_to_anchor=(0.53 ,0.53),
                 fancybox=False, columnspacing=0, framealpha=1, handlelength=1.2)
    # g.add_legend(ncol=2, )

    for ax in g.axes.flatten():
        # ax2.legend(ncol=2, loc='best', columnspacing=0) #bbox_to_anchor=(1.01, 1), borderaxespad=0)
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.set_ylim(bottom=40, top=90)
        # ax.set_xlim(left=0, right=1.6 * 10**9)
        ax.set_xscale('symlog')
        ax.set_xlim(left=0.04 * 10 ** 9, right=2 * 10 ** 9)

        ax.xaxis.set_minor_locator(ticker.SymmetricalLogLocator(base=10, linthresh=2, subs=[1.5, 2,3,4,5,6,8]))
        ax.xaxis.set_minor_formatter(b_formatter)
        ax.grid(which='minor', linestyle='--', color='#eeeeee')

        ax.xaxis.set_major_formatter(b_formatter)
        ax.tick_params(axis="x", which="both", rotation=90)
        #for text in ax.get_xminorticklabels():
        #    text.set_fontsize(10)

    plt.savefig(args.output, bbox_inches='tight')


def bd(args):
    exps = expman.gather(args.run).filter(args.filter)
    blink_metrics = exps.collect('test_pred/all_blink_roc_metrics.csv')
    blink_metrics = blink_metrics.iloc[3::4].rename({'0': 'auc'}, axis=1)
    

    import pdb; pdb.set_trace()


def metrics(args):
    exps = expman.gather(args.run).filter(args.filter)
    mask_metrics = exps.collect('test_pred/mask_metrics.csv')
    sns.lineplot(data=mask_metrics, x='thr', y='dice', hue='conv_type', size='grow_factor', style='num_filters')
    plt.savefig(args.output)
    

def log(args):
    exps = expman.gather(args.run).filter(args.filter)
    with PdfPages(args.output) as pdf:
        for exp_name, exp in sorted(exps.items()):
            print(exp_name)
            log = pd.read_csv(exp.path_to('log.csv'), index_col='epoch')
            train_cols = [c for c in log.columns if 'val' not in c]
            val_cols = [c for c in log.columns if 'val' in c]

            test_images = glob(os.path.join(exp.path_to('test_pred'), '*_samples.png'))

            fig = plt.figure(figsize=(14, 10))
            fig_shape = (2, 2) if test_images else (2, 1)
            ax1 = plt.subplot2grid(fig_shape, (0, 0))
            ax2 = plt.subplot2grid(fig_shape, (1, 0))

            # fig, (ax1, ax2) = plt.subplots(2, 1)
            log[train_cols].plot(ax=ax1)
            log[val_cols].plot(ax=ax2)
            ax1.legend(loc='center right', bbox_to_anchor=(-0.05, 0.5))
            ax2.legend(loc='center right', bbox_to_anchor=(-0.05, 0.5))

            if test_images:
                test_images = sorted(test_images)
                test_images = list(map(imread, test_images))
                max_w = max(i.shape[1] for i in test_images)
                pads = [((0,0), (0, max_w - i.shape[1]), (0, 0)) for i in test_images]
                test_images = np.concatenate([np.pad(i, pad) for i, pad in zip(test_images, pads)], axis=0)

                ax3 = plt.subplot2grid(fig_shape, (0, 1), rowspan=2)
                ax3.imshow(test_images)
                ax3.set_axis_off()

            log_plot_file = exp.path_to('log_plot.pdf')
            plt.suptitle(exp_name)
            plt.savefig(log_plot_file, bbox_inches='tight')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Show stuff')
    parser.add_argument('-f', '--filter', default={}, type=expman.exp_filter)
    subparsers = parser.add_subparsers()

    parser_log = subparsers.add_parser('log')
    parser_log.add_argument('run', default='runs/')
    parser_log.add_argument('-o', '--output', default='log_summary.pdf')
    parser_log.set_defaults(func=log)

    parser_metrics = subparsers.add_parser('metrics')
    parser_metrics.add_argument('run', default='runs/')
    parser_metrics.add_argument('-o', '--output', default='mask_metrics_summary.pdf')
    parser_metrics.set_defaults(func=metrics)

    parser_ee = subparsers.add_parser('ee')
    parser_ee.add_argument('run', default='runs/')
    parser_ee.add_argument('-o', '--output', default='ee_summary.pdf')
    parser_ee.set_defaults(func=ee)

    parser_bd = subparsers.add_parser('bd')
    parser_bd.add_argument('run', default='runs/')
    parser_bd.add_argument('-o', '--output', default='bd_summary.pdf')
    parser_bd.set_defaults(func=bd)

    args = parser.parse_args()
    args.func(args)

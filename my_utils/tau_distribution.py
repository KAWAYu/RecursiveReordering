#!/usr/bin/python3
# -*- coding: utf-8 -*-

import argparse
import codecs
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('base_tau', help='base tau file')
    parser.add_argument('btg_tau', help='btg tau file')
    parser.add_argument('rvnn_tau', help='rvnn tau file')
    parser.add_argument('--output', default='tau_distribution.pdf', type=str, help='output file name')
    return parser.parse_args()


def make_bars(data, bins=20):
    data_nums = [0] * bins
    for d in data:
        if d == 1.0:
            data_nums[-1] += 1
        else:
            data_nums[math.floor((d + 1) * 10)] += 1
    return data_nums


def make_tau_bleu(taus, bleus, bins=20):
    taus_num = [0 for _ in range(bins)]
    bleus_sum = [0 for _ in range(bins)]
    for t, b in zip(taus, bleus):
        if t == 1.0:
            taus_num[-1] += 1
            bleus_sum[-1] += b
        else:
            idx = math.floor((t + 1) * 10)
            taus_num[idx] += 1
            bleus_sum[idx] += b
    return taus_num, bleus_sum


if __name__ == '__main__':
    args = parse()
    with codecs.open(args.base_tau) as base_t, codecs.open(args.btg_tau) as btg_t, \
            codecs.open(args.rvnn_tau) as rvnn_t:
        base_taus = [float(t.strip()) for t in base_t]
        btg_taus = [float(t.strip()) for t in btg_t]
        rvnn_taus = [float(t.strip()) for t in rvnn_t]
    base_taus_num = make_bars(base_taus)
    btg_taus_num = make_bars(btg_taus)
    rvnn_taus_num = make_bars(rvnn_taus)
    base_taus_normed = [t / len(base_taus) for t in base_taus_num]
    btg_taus_normed = [t / len(btg_taus) for t in btg_taus_num]
    rvnn_taus_normed = [t / len(rvnn_taus) for t in rvnn_taus_num]

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.get_xaxis().get_major_formatter().set_useOffset(False)
    ax1.get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
    ax1.bar([(i-10)/10 for i in range(0, 20)], base_taus_normed, width=0.03, align="edge", linewidth=1, edgecolor="#000000", label="tau of w/o preordering")
    ax1.bar([(i-10+0.3)/10 for i in range(0, 20)], btg_taus_normed, width=0.03, align="edge", linewidth=1, edgecolor="#000000", label="tau of preordering with BTG")
    ax1.bar([(i-10+0.6)/10 for i in range(0, 20)], rvnn_taus_normed, width=0.03, align="edge", linewidth=1, edgecolor="#000000", label="tau of preordering with RvNN")
    ax1.set_ylabel("proportion", fontsize=24)
    ax1.set_xlabel(r"Kendall's $\tau$", fontsize=24)
    ax1.set_ylim(0, max(max(base_taus_normed), max(btg_taus_normed), max(rvnn_taus_normed)) + 0.05)
    ax1.tick_params(labelsize=18)
    ax1.legend(prop={'size': 18})
    fig.tight_layout()
    plt.savefig(args.output)

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import argparse
import codecs
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('base_tau', help='base tau file')
    parser.add_argument('btg_tau', help='btg tau file')
    parser.add_argument('rvnn_tau', help='rvnn tau file')
    parser.add_argument('base_bleu', help='base bleu file')
    parser.add_argument('btg_bleu', help='btg bleu file')
    parser.add_argument('rvnn_bleu', help='rvnn bleu file')
    parser.add_argument('--output', default='tau_histogram.pdf', type=str, help='output file name')
    return parser.parse_args()


def make_bars(data, bins=20):
    data_nums = [0] * bins
    for d in data:
        if d == 1.0:
            data_nums[-1] += 1
        else:
            data_nums[math.floor((d + 1) * 10)] += 1
    return data_nums


if __name__ == '__main__':
    args = parse()
    with codecs.open(args.base_tau) as base_t, codecs.open(args.btg_tau) as btg_t, \
            codecs.open(args.rvnn_tau) as rvnn_t, codecs.open(args.base_bleu) as base_b, \
            codecs.open(args.btg_bleu) as btg_b, codecs.open(args.rvnn_bleu) as rvnn_b:
        base_taus = [float(t.strip()) for t in base_t]
        btg_taus = [float(t.strip()) for t in btg_t]
        rvnn_taus = [float(t.strip()) for t in rvnn_t]
        base_bleus = [float(b.strip()) for b in base_b]
        btg_bleus = [float(b.strip()) for b in btg_b]
        rvnn_bleus = [float(b.strip()) for b in rvnn_b]
    base_taus_normed = [t / len(base_taus) for t in make_bars(base_taus)]
    btg_taus_normed = [t / len(btg_taus) for t in make_bars(btg_taus)]
    rvnn_taus_normed = [t / len(rvnn_taus) for t in make_bars(rvnn_taus)]

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.bar([(i-10)/10 for i in range(0, 20)], base_taus_normed, width=0.03, align="edge", linewidth=1, edgecolor="#000000", label="w/o preordering")
    ax1.bar([(i-10+0.3)/10 for i in range(0, 20)], btg_taus_normed, width=0.03, align="edge", linewidth=1, edgecolor="#000000", label="preordering with BTG")
    ax1.bar([(i-10+0.6)/10 for i in range(0, 20)], rvnn_taus_normed, width=0.03, align="edge", linewidth=1, edgecolor="#000000", label="preordering with RvNN")
    ax1.set_ylabel("proportion", fontsize=20)
    ax1.set_xlabel("w/o preordering", fontsize=20)
    ax1.set_ylim(0, max(max(base_taus_normed), max(btg_taus_normed), max(rvnn_taus_normed)) + 0.05)
    ax1.tick_params(labelsize=18)
    ax1.legend()
    fig.tight_layout()
    plt.savefig(args.output)

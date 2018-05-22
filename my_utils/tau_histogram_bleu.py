#!/usr/bin/python3
# -*- coding: utf-8 -*-

import argparse
import codecs
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True


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
            codecs.open(args.rvnn_tau) as rvnn_t, codecs.open(args.base_bleu) as base_b, \
            codecs.open(args.btg_bleu) as btg_b, codecs.open(args.rvnn_bleu) as rvnn_b:
        base_taus = [float(t.strip()) for t in base_t]
        btg_taus = [float(t.strip()) for t in btg_t]
        rvnn_taus = [float(t.strip()) for t in rvnn_t]
        base_bleus = [float(b.strip()) for b in base_b]
        btg_bleus = [float(b.strip()) for b in btg_b]
        rvnn_bleus = [float(b.strip()) for b in rvnn_b]
    base_taus_num, base_bleus_sum = make_tau_bleu(base_taus, base_bleus)
    btg_taus_num, btg_bleus_sum = make_tau_bleu(btg_taus, btg_bleus)
    rvnn_taus_num, rvnn_bleus_sum = make_tau_bleu(rvnn_taus, rvnn_bleus)
    base_taus_normed = [t / len(base_taus) for t in base_taus_num]
    btg_taus_normed = [t / len(btg_taus) for t in btg_taus_num]
    rvnn_taus_normed = [t / len(rvnn_taus) for t in rvnn_taus_num]
    base_bleus_normed = [b / t if t != 0 else 0.0 for b, t in zip(base_bleus_sum, base_taus_num)]
    btg_bleus_normed = [b / t if t != 0 else 0.0 for b, t in zip(btg_bleus_sum, btg_taus_num)]
    rvnn_bleus_normed = [b / t if t != 0 else 0.0 for b, t in zip(rvnn_bleus_sum, rvnn_taus_num)]

    print(base_bleus_normed)
    print(btg_bleus_normed)
    print(rvnn_bleus_normed)

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.bar([(i-10)/10 for i in range(0, 20)], base_taus_normed, width=0.03, align="edge", linewidth=1, edgecolor="#000000", label="tau of w/o preordering")
    ax1.bar([(i-10+0.3)/10 for i in range(0, 20)], btg_taus_normed, width=0.03, align="edge", linewidth=1, edgecolor="#000000", label="tau of preordering with BTG")
    ax1.bar([(i-10+0.6)/10 for i in range(0, 20)], rvnn_taus_normed, width=0.03, align="edge", linewidth=1, edgecolor="#000000", label="tau of preordering with RvNN")
    ax2 = ax1.twinx()
    ax2.scatter([(i - 10) / 10 for i in range(0, 20)], base_bleus_normed, 'o', label="bleu of w/o preordering")
    ax2.scatter([(i - 10) / 10 for i in range(0, 20)], btg_bleus_normed, 'o', label="bleu of preordering with BTG")
    ax2.scatter([(i - 10) / 10 for i in range(0, 20)], rvnn_bleus_normed, 'o', label="bleu of preordering with RvNN")
    ax1.set_ylabel("proportion", fontsize=20)
    ax2.set_ylabel("BLEU score", fontsize=20)
    ax1.set_xlabel(r"Kendall's $\tau$", fontsize=20)
    ax1.set_ylim(0, max(max(base_taus_normed), max(btg_taus_normed), max(rvnn_taus_normed)) + 0.05)
    ax1.tick_params(labelsize=18)
    ax1.legend()
    ax2.legend()
    fig.tight_layout()
    plt.savefig(args.output)

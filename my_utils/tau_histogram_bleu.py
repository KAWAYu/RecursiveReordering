#!/usr/bin/python3
#-*- coding: utf-8 -*-

import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def make_bars(data, bins=20):
    data_nums = [0] * bins
    for d in data:
        if d == 1.0:
            data_nums[-1] += 1
        else:
            data_nums[math.floor((d + 1) * 10)] += 1
    return data_nums


if __name__ == '__main__':
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax1.bar([(i-10)/10 for i in range(0, 20)], base_taus_normed, width=0.03, align="edge", linewidth=1, edgecolor="#000000", label="w/o preordering")
    ax1.bar([(i-10+0.3)/10 for i in range(0, 20)], btg_normed, width=0.03, align="edge", linewidth=1, edgecolor="#000000", label="preordering with BTG")
    ax1.bar([(i-10+0.6)/10 for i in range(0, 20)], taus_normed, width=0.03, align="edge", linewidth=1, edgecolor="#000000", label="preordering with RvNN")
    ax1.set_ylabel("proportion", fontsize=20)
    ax1.set_xlabel("w/o preordering", fontsize=20)
    ax1.set_ylim(0, 0.2)
    ax1.tick_params(labelsize=18)
    ax1.legend()
    fig.tight_layout()
    plt.savefig('tau_histogram.pdf')

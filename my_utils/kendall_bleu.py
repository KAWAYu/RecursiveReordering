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
    parser.add_argument('tau1', help='tau1 file')
    parser.add_argument('tau2', help='tau2 file')
    parser.add_argument('bleu1', help='bleu1 file')
    parser.add_argument('bleu2', help='bleu2 file')
    parser.add_argument('--fig_name', default="tau_bleu.pdf", type=str, help="figure name")
    return parser.parse_args()


def calc_corr_coef(xs, ys):
    """
    相関係数、分散を計算する関数
    :param xs: １つ目のデータ
    :param ys: ２つ目のデータ
    :return: 相関係数、１つ目の分散、２つ目の分散
    """
    avg_x = sum(xs) / len(xs)
    avg_y = sum(ys) / len(ys)
    numerator = sum((x - avg_x) * (y - avg_y) for x, y in zip(xs, ys))
    x_square = sum((x - avg_x) ** 2 for x in xs)
    y_square = sum((y - avg_y) ** 2 for y in ys)
    denominator = math.sqrt(x_square * y_square)
    return numerator / denominator, x_square / len(xs), y_square / len(ys)


if __name__ == '__main__':
    args = parse()
    with codecs.open(args.tau1, 'r', 'utf-8') as tau1, codecs.open(args.tau2, 'r', 'utf-8') as tau2, \
            codecs.open(args.bleu1, 'r', 'utf-8') as bleu1, codecs.open(args.bleu2, 'r', 'utf-8') as bleu2:
        diff_taus = [float(tau_line2.strip()) - float(tau_line1.strip())
                     for tau_line1, tau_line2 in zip(tau1, tau2)]
        diff_bleus = [float(bleu_line2.strip()) - float(bleu_line1.strip())
                      for bleu_line1, bleu_line2 in zip(bleu1, bleu2)]
    plt.figure()
    plt.scatter(diff_taus, diff_bleus)
    plt.xlabel("diff kendall's tau")
    plt.ylabel("diff bleu")
    plt.legend()
    plt.savefig(args.fig_name)
    print("corr_coef: %f, variance of BLEU: %f, variance of tau: %f" % calc_corr_coef(diff_bleus, diff_taus))

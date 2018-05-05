#!/usr/bin/python3
# -*- coding: utf-8 -*-

import argparse
import codecs
import numpy as np
import matplotlib.pyplot as plt


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('tau1', help='tau1 file')
    parser.add_argument('tau2', help='tau2 file')
    parser.add_argument('bleu1', help='bleu1 file')
    parser.add_argument('bleu2', help='bleu2 file')
    return parser.parse_args()


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
    plt.show()

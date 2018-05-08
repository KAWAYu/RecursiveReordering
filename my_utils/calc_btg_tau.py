#!/usr/bin/python3
# -*- coding: utf-8 -*-

import argparse
import gzip
import codecs
import re
from my_utils import cc_rank


def parse():
    parser = argparse.ArgumentParser(
        usage="python3 calc_btg_tau.py <BTG's order file> <Alignment file> <OUTPUT>",
        description="calculate btg's tau"
    )

    parser.add_argument("btg_order_file", help="BTG's order file path")
    parser.add_argument("alignment_file", help="Alignment file path")
    parser.add_argument("output", help="Output file path")
    parser.add_argument("--base", dest="base", action="store_true")
    parser.set_defaults(base=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse()
    taus = []
    base_taus = []

    with codecs.open(args.btg_order_file, 'r', 'utf-8') as btg, \
            gzip.open(args.alignment_file, 'r') as af:
        for bline, _ in zip(btg, af):
            t_idx = []
            b_idx = bline.strip().split()
            sline = af.readline().strip().decode('utf-8').split()
            tline = af.readline().strip().decode('utf-8')
            ttokens = re.split('\(\{|\}\)', tline)[:-1]
            aligns = []
            for i in range(len(ttokens) // 2):
                aligns.append([j for j in ttokens[i * 2].strip().split()])
            for b_i in b_idx:
                for k, align in enumerate(aligns):
                    if str(int(b_i) + 1) in align:
                        t_idx.append(k)
            base_align = []
            for align in aligns:
                base_align += map(int, align)
            base_taus.append(cc_rank.kendall_tau(base_align))
            taus.append(cc_rank.kendall_tau(t_idx))

    with codecs.open(args.output, 'w', 'utf-8') as out_file:
        for tau in taus:
            print(tau, file=out_file)

    with codecs.open(args.output + '.base', 'w', 'utf-8') as out_file:
        for tau in base_taus:
            print(tau, file=out_file)

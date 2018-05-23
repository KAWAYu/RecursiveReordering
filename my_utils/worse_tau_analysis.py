#!/usr/bin/python3
# -*- coding; utf-8 -*-

import argparse
import gzip


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('tree_file', help='tree file')
    parser.add_argument('base_tau', help='base tau file')
    parser.add_argument('btg_tau', help='btg tau file')
    parser.add_argument('btged_file', help='reordered sentence file')
    parser.add_argument('alignment', help='alignment file')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse()
    with open(args.tree_file) as tree_f, open(args.base_tau) as base_f, open(args.btg_tau) as btg_f, \
            open(args.btged_file) as btged_f, gzip.open(args.alignment, 'r') as align:
        for tree_line, base_tau, btg_tau, btged_line, _ in zip(tree_f, base_f, btg_f, btged_f, align):
            align_s = align.readline().strip().decode('utf-8')
            align_t = align.readline().strip().decode('utf-8')
            if float(base_tau) > float(btg_tau):
                print(tree_line.strip())
                print(btged_line.strip())
                print(align_s)
                print(align_t)
                print()

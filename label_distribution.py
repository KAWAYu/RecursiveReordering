#!/usr/bin/python3
# -*- coding: utf-8 -*-

import argparse
import codecs
import collections
import gzip
import re

from Recursive_util_pos import EnjuXmlParser, STreeParser, kendall_tau


def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('tree_file', help="tree file path")
    parser.add_argument('alignment_file', help="alignment file path")
    parser.add_argument('--tree_type', default="enju", help="tree type (enju or s)")
    parser.add_argument('--label', '-l', default=2, type=int, help="the label num")

    return parser.parse_args()


def read_tree_file(tfile, align_file, tt, l_dict):
    with codecs.open(tfile, 'r', 'utf-8') as fin, gzip.open(align_file, 'r') as afile:
        for line in fin:
            line = line.strip()
            afile.readline()
            e_words = afile.readline().strip().decode('utf-8').split()
            f_line = afile.readline().strip().decode('utf-8')
            f_words = re.split('\(\{|\}\)', f_line)[:-1]
            tmp_vocab_dict = collections.defaultdict(lambda: 0)
            e_wordlist = []
            # 原言語側の処理（単語がかぶっている時は単語前に数字をつけて区別）
            for e_word in e_words:
                tmp_vocab_dict[e_word] += 1
                e_wordlist.append(str(tmp_vocab_dict[e_word]) + '_' + e_word)

            f_word_dst = []
            align = []
            for j in range(len(f_words) // 2):
                # NULLアライメントは考慮しないのでfor文は1から
                f_word = f_words[2 * j]  # 目的言語側の単語
                f_align = f_words[2 * j + 1].strip().split()  # 目的言語のアライメント先
                f_word_dst.append((f_word, [e_wordlist[int(k) - 1] for k in f_align]))
                align.extend([int(k) for k in f_align])

            if tt == "enju":
                tree = EnjuXmlParser(line)
            elif tt == "s":
                tree = STreeParser(line)
            tree = tree.parse(tree.root)
            if tree['status'] == 'failed':
                continue
            convert_tree(tree, e_wordlist, f_word_dst, 0, kendall_tau(align), l_dict)


def convert_tree(node, elist, fdst_list, j, tau, l_dict):
    if node['tag'] == 'sentence':
        span = (j, j)
        for child in node['children']:
            span = convert_tree(child, elist, fdst_list, span[1], tau, l_dict)
    elif node['tag'] == 'cons':
        assert len(node['children']) == 1 or len(node['children']) == 2
        if len(node['children']) == 1:
            return convert_tree(node['children'][0], elist, fdst_list, j, tau, l_dict)
        else:
            left_span = convert_tree(node['children'][0], elist, fdst_list, j, tau, l_dict)
            right_span = convert_tree(node['children'][1], elist, fdst_list, max(left_span), tau, l_dict)
            span_min, span_max = min(left_span), max(right_span)
            tmp_e_list = [elist[i] for i in range(span_min - 1)] + [elist[i - 1] for i in left_span] \
                + [elist[i - 1] for i in right_span] + [elist[i] for i in range(span_max, len(elist))]
            align = []
            for f_w_dst in fdst_list:
                for e_dst in f_w_dst[1]:
                    if e_dst in tmp_e_list:
                        align.append(tmp_e_list.index(e_dst))
            tmp_tau_1 = kendall_tau(align)
            tmp_e_list = [elist[i] for i in range(span_min - 1)] + [elist[i - 1] for i in right_span] \
                + [elist[i - 1] for i in left_span] + [elist[i] for i in range(span_max, len(elist))]
            align = []
            for f_w_dst in fdst_list:
                for e_dst in f_w_dst[1]:
                    if e_dst in tmp_e_list:
                        align.append(tmp_e_list.index(e_dst))
            tmp_tau_2 = kendall_tau(align)
            if tmp_tau_2 > tmp_tau_1:
                if len(l_dict) == 3 and tmp_tau_1 == tmp_tau_2:
                    l_dict["Dont care"] += 1
                else:
                    l_dict["Inverted"] += 1
                span_list = right_span + left_span
            else:
                l_dict["Straight"] += 1
                span_list = left_span + right_span
            return span_list
    elif node['tag'] == 'tok':
        return [j+1]


if __name__ == '__main__':
    args = parse()
    label_dict = None
    if args.label == 2:
        label_dict = {"Straight": 0, "Inverted": 0}
    elif args.label == 3:
        label_dict = {"Straight": 0, "Inverted": 0, "Dont cate": 0}
    read_tree_file(args.tree_file, args.alignment_file, args.tree_type, label_dict)
    total = sum(v for _, v in label_dict.items())
    print(' '.join("%s %.4f" % (k, v / total) for k, v in label_dict.items()))

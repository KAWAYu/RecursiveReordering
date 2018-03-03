#!/usr/bin/python3
# -*- coding: utf-8 -*-

import re
import gzip
import argparse
import codecs
from collections import defaultdict


def parse():
    parser = argparse.ArgumentParser(
        description = 'Preprocess script for btg'
    )
    parser.add_argument('alignment_file', help='alignment file path')
    parser.add_argument('--sout', help='output of source file')
    parser.add_argument('--tout', help='output of target file')
    parser.add_argument('--align', help='output of alignment file')
    args = parser.parse_args()

    return args


def ngram(sentence, n=2):
    return [sentence[i:i+n] for i in range(len(sentence) - n + 1)]


if __name__ == '__main__':
    args = parse()
    sentences = []
    btg_sentences = []
    gram5_el = defaultdict(lambda: [])
    gram5 = defaultdict(lambda: [])
    i = 0
    with gzip.open(args.alignment_file, 'rb') as f_align, codecs.open('source.tmp', 'w', 'utf-8') as s_tmp, \
            codecs.open('target.tmp', 'w', 'utf-8') as t_tmp, codecs.open(args.align, 'w', 'utf-8') as align_out:
        print("Reading alignment file...")
        while f_align.readline():
            source = f_align.readline().decode('utf-8').strip().split()
            target = f_align.readline().decode('utf-8').strip()
            target_words = [w for i, w in enumerate(re.split('\(\{|\}\)', target)[:-1]) if i % 2 == 0]
            target_aligns = [map(int, a.split()) for i, a in enumerate(re.split('\(\{|\}\)', target)[:-1]) if i % 2 == 1]
            # 3単語以上50単語以下の場合は放置
            if len(source) <= 2 or len(source) >= 51 or len(target_words) <= 2 or len(target_words) >= 51:
                continue
            # アライメントが3つ以上かつ原言語の単語の数の半分以上のものだけを保持
            if len(target_aligns) <= 2 or len(target_aligns) < len(target_words) // 2:
                continue
            # すでに5gramがある場合はその文をスキップ(前の方がいい文なので残す（いいのか？）)
            if any(_g[0] in gram5 and _g[1:] in gram5[_g[0]] for _g in ngram(source, n=5)):
                continue
            else:
                # 辞書に5gramを登録
                for _g in ngram(source, n=5):
                    if _g[0] not in gram5 or _g[1:] not in gram5[_g[0]]:
                        gram5[_g[0]].append(_g[1:])
                i += 1
                if i % 1000 == 0:
                    print("read %d lines..." % i)
                print("# Sentence Pair %d:" % (i + 1), file=align_out)
                print(source.strip(), file=align_out)
                print(target.strip(), file=align_out)

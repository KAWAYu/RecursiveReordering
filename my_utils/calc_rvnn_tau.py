#!/usr/bin/python3
# -*- coding: utf-8 -*-


import xml.etree.ElementTree as ET
from chainer import Chain, Variable, serializers
import chainer.links as L
import chainer.functions as F
import re
import codecs
import collections
import gzip
import argparse
import numpy as np
import pickle

from my_utils import cc_rank


class EnjuXmlParser(object):
    """
    enjuのxml出力を読み込むクラス
    各々１本の木を保有する
    """
    def __init__(self, sentence):
        self.root = ET.fromstring(sentence)

    def parse(self, node, root=True):
        children = []
        this_node = {}

        if root and node.attrib['parse_status'] != 'success':
            # enjuが構文解析失敗している場合「失敗」を返す
            this_node['status'] = 'failed'
            return this_node

        # 子の処理
        for c in node:
            children.append(self.parse(c, root=False))

        tag = node.tag
        if tag == 'sentence':
            nodeid = node.attrib['id']
            status = node.attrib['parse_status']
            this_node['status'] = status
        elif tag == 'cons':
            nodeid = node.attrib['id']
            cat = node.attrib['cat']
            head = node.attrib['head']
            this_node['cat'] = cat
            this_node['head'] = head
            if node.text:
                this_node['text'] = node.text
        elif tag == 'tok':
            nodeid = node.attrib['id']
            cat = node.attrib['cat']
            pos = node.attrib['pos']
            text = node.text
            this_node['cat'] = cat
            this_node['pos'] = pos
            this_node['text'] = text
        this_node['id'] = nodeid
        this_node['tag'] = tag
        this_node['children'] = children
        return this_node


class STreeParser(object):
    def __init__(self, sentence):
        self.root = self.init_parse(sentence)

    def init_parse(self, parse_snt):
        phrases = [[]]
        tmp_str = ""
        for c in parse_snt.strip():
            if c == '(':
                phrases.append([])
            elif c == ' ':
                if tmp_str:
                    phrases[-1].append(tmp_str)
                    tmp_str = ""
            elif c == ')':
                if tmp_str:
                    phrases[-1].append(tmp_str)
                    phrases[-2].append(phrases.pop())
                    tmp_str = ""
                else:
                    phrases[-2].append(phrases.pop())
            else:
                tmp_str += c
        return phrases[0][0]

    def parse(self, node, root=True):
        this_node = {}
        if root:  # 根ノード
            if node == [[]]:
                return {'status': 'parse_error'}
            this_node['status'] = 'success'
            this_node['tag'] = 'sentence'
            this_node['children'] = [self.parse(node[-1], root=False)]
        elif len(node) == 2:  # 子が二つの時
            if isinstance(node[1], str):  # 葉ノード
                this_node['tag'] = 'tok'
                this_node['pos'] = node[0]
                this_node['text'] = node[1]
            else:  # 節ノード
                this_node['tag'] = 'cons'
                this_node['children'] = [self.parse(node[1], root=False)]
                this_node['cat'] = node[0]
        else:  # 子が三つの時
            this_node['tag'] = 'cons'
            this_node['cat'] = node[0]
            left_node = self.parse(node[1], root=False)
            right_node = self.parse(node[2], root=False)
            this_node['children'] = [left_node, right_node]
        return this_node


class RecursiveNet(Chain):
    def __init__(self, n_vocab, n_pos, n_embed, n_pos_embed, n_units, n_labels):
        super(RecursiveNet, self).__init__(
            pos_emb=L.EmbedID(n_pos, n_pos_embed),
            emb=L.EmbedID(n_vocab, n_embed),
            pel=L.Linear(n_embed + n_pos_embed, n_units),
            l=L.Linear(n_units*2 + n_pos_embed, n_units),
            w=L.Linear(n_units, n_labels),
        )

    def leaf(self, x, p):
        return F.relu(self.pel(F.concat((F.relu(self.emb(x)), F.relu(self.pos_emb(p))))))

    def node(self, left, right, p):
        return F.relu(self.l(F.concat((F.concat((left, right)), F.relu(self.pos_emb(p))))))

    def label(self, v):
        return self.w(v)


def read_tree_data(tree_file_path, align_file_path, vocab, cat_vocab):
    trees = []
    with codecs.open(tree_file_path, 'r', 'utf-8') as tree_file_path, gzip.open(align_file_path, 'r') as align_file:
        for i, tree_line in enumerate(tree_file_path):
            if (i+1) % 1000 == 0:
                print("%d lines have been read..." % (i+1))
            tree_line = tree_line.strip()
            align_file.readline()
            # 原言語の読み込み(3n+2行目)
            e_words = align_file.readline().strip().decode('utf-8').split()
            # 目的言語（アライメント先）の読み込み(3n行目)
            f_line = align_file.readline().strip().decode('utf-8')
            f_words = re.split('\(\{|\}\)', f_line)[:-1]
            tmp_vocab_dict = collections.defaultdict(lambda: 0)

            e_wordlist = []
            # 原言語側の処理（単語がかぶっている時は単語前に数字をつけて区別）
            for e_word in e_words:
                tmp_vocab_dict[e_word] += 1
                e_wordlist.append(str(tmp_vocab_dict[e_word]) + '_' + e_word)

            f_word_dst = []
            align = []
            for j in range(1, len(f_words)//2):
                # NULLアライメントは考慮しないのでfor文は1から
                f_word = f_words[2*j]  # 目的言語側の単語
                f_align = f_words[2*j+1].strip().split()  # 目的言語のアライメント先
                f_word_dst.append((f_word, list(map(int, f_align))))
                align.extend([int(k) for k in f_align])

            tree = EnjuXmlParser(tree_line)
            tree = tree.parse(tree.root)
            if tree['status'] == 'failed':
                continue
            trees.append(convert_tree(vocab, tree, e_wordlist, f_word_dst, 0, cc_rank.kendall_tau(align), cat_vocab))
    return trees


def convert_tree(vocab, node, e_list, f_dst_list, j, tau, cat_vocab):
    """
    :param vocab: 単語->idの辞書
    :param node: 今のノード
    :param e_list: 原言語のリスト ex)["I", "live", "in", "London"]
    :param f_dst_list: 目的言語がどの原言語に対応しているかのタプルのリスト
                       ex)[("私は", [1]), ("ロンドン", [4]), ("に", [3]), ("住んで", [2]), ("いる", [])]
    :param j: 被覆している単語の右端のインデックス
    :param tau: 今のケンダールのτ
    :return node['tag'] == 'sentence'なら木のルートノード
            それ以外なら部分木のスパンと部分木のルートノード
    """
    if node['tag'] == 'sentence':
        children = []
        span = (j,j)
        for child in node['children']:
            span, child_node = convert_tree(vocab, child, e_list, f_dst_list, span[1], tau, cat_vocab)
            children.append(child_node)
        return {'tag': 'sentence', 'children': children}
    elif node['tag'] == 'cons':
        assert len(node['children']) == 1 or len(node['children']) == 2
        if len(node['children']) == 1:
            return convert_tree(vocab, node['children'][0], e_list, f_dst_list, j, tau, cat_vocab)
        else:
            swap = 0
            left_span, left_node = convert_tree(vocab, node['children'][0], e_list, f_dst_list, j, tau, cat_vocab)
            right_span, right_node = convert_tree(vocab, node['children'][1], e_list, f_dst_list, max(left_span), tau, cat_vocab)
            # 並び替え候補の作成
            span_min, span_max = min(left_span), max(right_span)
            # 並び替えなしの時の単語の順番
            tmp_e_list = [e_list[i] for i in range(span_min - 1)] + [e_list[i-1] for i in left_span] \
                + [e_list[i-1] for i in right_span] + [e_list[i] for i in range(span_max, len(e_list))]
            align = []
            for f_w_dst in f_dst_list:
                for e_dst in f_w_dst[1]:
                    if e_dst in tmp_e_list:
                        align.append(tmp_e_list.index(e_dst))
            tmp_tau_1 = cc_rank.kendall_tau(align)
            # 並び替えあり
            tmp_e_list = [e_list[i] for i in range(span_min - 1)] + [e_list[i-1] for i in right_span] \
                + [e_list[i-1] for i in left_span] + [e_list[i] for i in range(span_max, len(e_list))]
            align = []
            for f_w_dst in f_dst_list:
                for e_dst in f_w_dst[1]:
                    if e_dst in tmp_e_list:
                        align.append(tmp_e_list.index(e_dst))
            tmp_tau_2 = cc_rank.kendall_tau(align)
            if tmp_tau_2 > tmp_tau_1:
                swap = 1
                span_list = right_span + left_span
            else:
                span_list = left_span + right_span
            cat_id = cat_vocab[node['cat']] if node['cat'] in cat_vocab else cat_vocab['<UNK>']
            return span_list, {'label': swap, 'node': (left_node, right_node), 'cat': node['cat'], 'cat_id': cat_id}
    elif node['tag'] == 'tok':
        t = vocab[node['text'].lower()] if node['text'].lower() in vocab else vocab['<UNK>']
        e_word = e_list[j]
        in_align = False
        for i, (_, e_dst) in enumerate(f_dst_list):
            if j+1 in e_dst:
                in_align = True
                break
        cat_id = cat_vocab[node['pos']] if node['pos'] in cat_vocab else cat_vocab['<UNK>']
        return [j+1], {'node': t, 'text': node['text'], 'alignment': i+1 if in_align else 0, 'pos': node['pos'], 'cat_id': cat_id}


def parse():
    parser = argparse.ArgumentParser(
        description='Reordering with Recursive Neural Network',
        usage='\n %(prog)s {train, test} filepath alignmentfile [options] reorderfile'
    )
    parser.add_argument('filepath', help='training file path')
    parser.add_argument('alignmentfile', help='alignment file(format ".gz")')
    parser.add_argument('vocab_pkl', help='vocabulary pickle file')
    parser.add_argument('model', help='model name')
    parser.add_argument('--unit', '-u', default=200, type=int, help='number of units')
    parser.add_argument('--label', '-l', default=2, type=int, help='number of labels')
    parser.add_argument('--embed_size', '-emb', default=200, type=int, help='number of embedding size')
    parser.add_argument('--pos_embed_size', '-pemb', default=200, type=int, help='number of pos-tag embedding size')
    parser.add_argument('--output', default="rvnn_tau.txt", type=str, help="rvnn's tau output file")
    return parser.parse_args()


def traverse(model, node, root=True):
    if root:
        alignment = []
        for child in node['children']:
            _, child_align = traverse(model, child, root=False)
            alignment += child_align
        return alignment
    elif 'text' in node:
        alignment = [node['alignment']] if node['alignment'] else []
        x = Variable(np.array([node['node']], dtype=np.int32))
        p = Variable(np.array([node['cat_id']], dtype=np.int32))
        v = model.leaf(x, p)
        return v, alignment
    else:
        left_node, right_node = node['node']
        left, left_align = traverse(model, left_node, root=False)
        right, right_align = traverse(model, right_node, root=False)
        p = Variable(np.array([node['cat_id']], dtype=np.int32))
        v = model.node(left, right, p)
        y = model.label(v)
        pred_label = y.data.argmax(1)
        if pred_label[0] == 0:
            alignment = left_align + right_align
        else:
            alignment = right_align + left_align
        return v, alignment


if __name__ == '__main__':
    args = parse()
    taus = []
    vocab = pickle.load(open(args.vocab_pkl, 'rb'))
    cat_vocab = pickle.load(open(args.vocab_pkl + '.pos', 'rb'))
    trees = read_tree_data(args.filepath, args.alignmentfile, vocab, cat_vocab)
    model = RecursiveNet(len(vocab), len(cat_vocab), args.embed_size, args.pos_embed_size, args.unit, args.label)
    serializers.load_hdf5(args.model, model)
    for i, tree in enumerate(trees):
        if (i + 1) % 1000 == 0:
            print("processed %d lines" % (i + 1))
        tree_align = traverse(model, tree)
        taus.append(cc_rank.kendall_tau(tree_align))
    with codecs.open(args.output, 'w', 'utf-8') as out:
        for tau in taus:
            print(tau, file=out)

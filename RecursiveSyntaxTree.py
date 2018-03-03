#!/usr/bin/python3
# -*- coding: utf-8 -*-

import argparse
import codecs
import gzip
import numpy as np
import time
import re

import chainer
from chainer import cuda, optimizers, Variable, serializers, Chain
import chainer.functions as F
import chainer.links as L


xp = np


class RecursiveNet(Chain):
    def __init__(self, vocab_size, embed_size, hidden_size, label_size):
        super(RecursiveNet, self).__init__(
            emb = L.EmbedID(vocab_size, embed_size),
            e = L.Linear(embed_size, hidden_size),
            d = L.Linear(hidden_size * 2, label_size),
            l = L.Linear(hidden_size * 2, hidden_size),
            w = L.Linear(hidden_size, label_size),
        )

    def leaf(self, x):
        """
        葉ノードのための関数。単語エンベッディング。
        """
        return F.relu(self.emb(x))

    def detect_node(self, left, right):
        """
        ノード対から結合するスコアを計算するための関数
        """
        return F.relu(self.d(F.concat(left, right)))

    def concat_node(self, left, right):
        """
        ノード対から親ノードのベクトルを作る関数
        """
        return F.relu(self.l(F.concat(left, right)))

    def label(self, node):
        """
        ノードからラベルを推定する関数
        """
        return self.w(node)


class Leaf(object):
    """
    葉ノードを表すクラス
    """
    def __init__(self, w, a):
        self.word = w
        self.alignment = a
        self.swap = 0

    def __repr__(self):
        return "<class Leaf: word " + self.word + ", alignment " + str(self.alignment) + " /Leaf>"


class Node(object):
    """
    節ノードを表すクラス
    """
    def __init__(self, left, right, label, swap):
        self.label = label
        self.left = left
        self.right = right
        self.swap = swap

    def __repr__(self):
        if self.label == "Straight":
            return "<class Node: [" + str(self.left) + ", " + str(self.right) + "] /Node>"
        else:
            return "<class Node: [" + str(self.right) + ", " + str(self.left) + "] /Node>"

class Nodes(object):
    """
    節ノード集合
    """
    def __init__(self, nodes):
        self.nodes = nodes

    def __repr__(self):
        return "<class Nodes: [" + ', '.join(str(node) for node in self.nodes) + "] /Nodes>"

    def add(node):
        self.nodes.append(node)


def parse():
    parser = argparse.ArgumentParser(
        description = 'Automatically constructing tree structure with recursive neural network',
        usage = '\n %(prog)s output_format alignmentfile [options]'
            '\n %(prog)s -h'
    )

    parser.add_argument('output_format', help='output_format [text, sequence]')
    parser.add_argument('alignmentfile', help='alignment file(format ".gz")')
    parser.add_argument('--vocab_pkl', default="", type=str, help='vocabulary pickle file')
    parser.add_argument('--vocab_size', '-vs', default=-1, type=int,
        help='the max number of vocabulary')
    parser.add_argument('--epoch', '-e', default=100, type=int,
        help='number of epoch to train')
    parser.add_argument('--hidden_size', '-hs', default=100, type=int,
        help='number of units')
    parser.add_argument('--embed_size', '-es', default=100, type=int,
        help='number of embedding size')
    parser.add_argument('--batchsize', '-b', default=128, type=int,
        help='size of minibatch')
    parser.add_argument('--label', '-l', default=2, type=int,
        help='number of labels')
    parser.add_argument('--evalinterval', '-p', default=5, type=int,
        help='number of epochs per evaluation')
    parser.add_argument('--gpus', '-g', default=-1, type=int,
        help='number of gpu you want to use')
    parser.add_argument('--model', '-m', default=-1, type=int,
        help='appoint the epoch if you start from previous training')
    parser.add_argument('--img_name', default="", type=str,
        help='image\'s name of learning curve')
    parser.add_argument('--optimize', default='adam', type=str,
        help='optimizer for neural network(default: Adam)')
    parser.add_argument('--visualize', '-v', dest='visualize', action='store_true')
    parser.set_defaults(visualize=False)
    args = parser.parse_args()

    return args


def print_message(msg):
    """
    現在時刻とともに引数を表示する関数
    :params msg: 表示したいメッセージ
    """
    print(time.asctime() + ': ' + msg)


def cky(leaves):
    """
    CKYアルゴリズムを模倣して木の構築を行う
    A -> BCのようなルールはないので全てのノードで候補があるため、各ノードで順位相関係数が高くなるように枝刈りを行う
    """
    last_node = None
    # 最後が記号(.?)の時は途中で結合して欲しくないので退避
    if leaves[-1].word in [".", "?"]:
        leaves, last_node = leaves[:-1], leaves[-1]
    trees = [[[n]] for n in leaves] # 葉ノードの構築
    len_leaves = len(leaves)
    for d in range(1, len_leaves):
        for i in range(len_leaves - d):
            nodes = []
            for j in range(len(trees[i])): # 被覆するスパンに対して全探索
                # 候補ノードの作成　下の場合分けは不要なので後で消す
                if len(trees[i+j+1]) == len(trees[i]):
                    nodes += make_candidate(trees[i][j], trees[i+j+1][-j-1])
                else:
                    nodes += make_candidate(trees[i][j], trees[i+j+1][-j-1 + (len(trees[i]) - len(trees[i+j+1]))])
            # 順位相関係数が高いもの
            max_tau = max(kendall_tau(flatten_node(n)) for n in nodes)
            nodes = [n for n in nodes if kendall_tau(flatten_node(n)) == max_tau]
            # 交換回数が少ないもの
            min_swap = min(n.swap for n in nodes)
            nodes = [n for n in nodes if n.swap == min_swap]
            trees[i].append(Nodes(nodes))
    ts = trees[0][-1]
    # 交換回数が最小のもの
    #min_swap = min(t.swap for t in ts)
    #trees = [t for t in ts if t.swap == min_swap]
    return ts


def make_candidate(left_nodes, right_nodes):
    nodes = []
    if isinstance(left_nodes, Nodes):
        left_nodes = left_nodes.nodes
    if isinstance(right_nodes, Nodes):
        right_nodes = right_nodes.nodes
    for left_node in left_nodes:
        for right_node in right_nodes:
            left_align = flatten_node(left_node)
            right_align = flatten_node(right_node)
            if kendall_tau(left_align + right_align) < kendall_tau(right_align + left_align):
                nodes.append(Node(left_node, right_node, "Inverted", left_node.swap + right_node.swap + 1))
            else:
                nodes.append(Node(left_node, right_node, "Straight", left_node.swap + right_node.swap))
    return nodes


def flatten_node(node):
    if isinstance(node, Leaf):
        return node.alignment
    elif isinstance(node, Node):
        if node.label == "Inverted":
            return flatten_node(node.right) + flatten_node(node.left)
        elif node.label == "Straight":
            return flatten_node(node.left) + flatten_node(node.right)
    elif isinstance(node, Nodes):
        tmp_align = []
        for n in Nodes.nodes:
            tmp_align += flatten_node(n)
        return tmp_align


def traverse(leaves, candidate_trees):
    while leaves != 1:
        pass


def kendall_tau(alignment):
    """
    ケンダールのτを計算する関数
    :params alignment: ソース側のアライメント
    """
    c = 0
    if len(alignment) == 0:
        return 0.0
    elif len(alignment) == 1:
        return 1.0
    else:
        for i in range(len(alignment) - 1):
            for j in range(i + 1, len(alignment)):
                if alignment[i] <= alignment[j]:
                    c += 1
    return 2 * c / (len(alignment) * (len(alignment) - 1)) - 1


def fuzzy_reordering_score(alignment):
    """
    Fuzzy Reordering Scoreを計算する関数
    :params alignment: ソース側のアライメント
    """
    B = 0
    B += 1 if alignment[0] == 1 else 0
    B += 1 if alignment[-1] == max(alignment) else 0
    for i in range(len(alignment[:-1])):
        if a[i] == a[i+1] or a[i] + 1 == a[i+1]:
            B += 1
    return B / (len(alignment) + 1)


def data_prepare(args):
    """
    ファイルから入力データと木構造を作る関数
    :param args: プログラムの引数。ファイルパスとかとかなどなど。
    """
    trees = []
    print_message("Reading data...")
    with gzip.open(args.alignmentfile, 'rb', 'utf-8') as align_file:
        for line in align_file:
            source = align_file.readline().strip().decode('utf-8')  # 原言語側の文
            target = align_file.readline().strip().decode('utf-8')  # 目的言語側の文
            tree = [Leaf(w, []) for w in source.split()]
            target_words_align = re.split('\(\{|\}\)', target)[2:-1]
            target_align = [a for i, a in enumerate(target_words_align) if i % 2 == 1]
            for i, a in enumerate(target_align):
                if a.strip():
                    for _a in a.strip().split():
                        tree[int(_a) - 1].alignment.append(i + 1)
            trees.append(tree)
    print_message("Construct Tree...")
    for tree in trees:
        # yield tree, construct_trees(tree)
        yield tree, cky(tree)


def main():
    import pprint
    args = parse()
    print_message("Prepare training data...")
    for c_t in data_prepare(args):
        pprint.pprint(c_t)
        input()


if __name__ == '__main__':
    main()

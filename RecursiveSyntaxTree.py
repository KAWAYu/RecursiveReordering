#!/usr/bin/python3
# -*- coding: utf-8 -*-

import argparse
import codecs
import gzip
import numpy as np
import time
import re
import random

import chainer
from chainer import cuda, optimizers, serializers, Chain, Variable
import chainer.functions as chainF
import chainer.links as chainL


xp = np


class RecursiveNet(Chain):
    def __init__(self, vocab_size, embed_size, hidden_size, label_size):
        super(RecursiveNet, self).__init__()
        with self.init_scope():
            self.emb = chainL.EmbedID(vocab_size, embed_size)
            self.e = chainL.Linear(embed_size, hidden_size)
            self.d = chainL.Linear(hidden_size * 2, 1)
            self.l = chainL.Linear(hidden_size * 2, hidden_size)
            self.w = chainL.Linear(hidden_size, label_size)

    def leaf(self, x):
        """
        葉ノードのための関数。単語エンベッディング。
        """
        return chainF.tanh(self.e(chainF.tanh(self.emb(x))))

    def detect_node(self, left, right):
        """
        ノード対から結合するスコアを計算するための関数
        """
        return chainF.sigmoid(self.d(chainF.concat((left, right))))

    def concat_node(self, left, right):
        """
        ノード対から親ノードのベクトルを作る関数
        """
        ret = chainF.tanh(self.l(chainF.concat((left, right))))
        return ret

    def label(self, node):
        """
        ノードからラベルを推定する関数
        """
        return self.w(node)


class Leaf(object):
    """
    葉ノードを表すクラス
    """
    def __init__(self, w, a, idx):
        self.word = w
        self.index = idx
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

    def add(self, node):
        self.nodes.append(node)


def parse():
    parser = argparse.ArgumentParser(
        description='Automatically constructing tree structure with recursive neural network',
        usage='\n %(prog)s output_format alignmentfile [options]'
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
    A -> BCのようなルールはなく全てのノードで候補があるため、各ノードで順位相関係数が高くなるように枝刈りを行う
    """
    print_message("Construct Tree...")
    last_node = None
    # 最後が記号(.?)の時は途中で結合して欲しくないので退避
    if leaves[-1].word in [".", "?"]:
        leaves, last_node = leaves[:-1], leaves[-1]
    trees = [[[n]] for n in leaves]  # 葉ノードの構築
    len_leaves = len(leaves)
    for d in range(1, len_leaves):
        for i in range(len_leaves - d):
            nodes = []
            for j in range(len(trees[i])):  # 被覆するスパンに対して全探索
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
    if last_node:
        ts = Nodes([Node(n, last_node, "Straight", n.swap) for n in ts.nodes])
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
        for n in node.nodes:
            tmp_align += flatten_node(n)
        return tmp_align


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
    return 2 * c / (len(alignment) * (len(alignment)-1) / 2) - 1


def fuzzy_reordering_score(alignment):
    """
    Fuzzy Reordering Scoreを計算する関数
    :params alignment: ソース側のアライメント
    """
    B = 0
    B += 1 if alignment[0] == 1 else 0
    B += 1 if alignment[-1] == max(alignment) else 0
    for i in range(len(alignment[:-1])):
        if alignment[i] == alignment[i+1] or alignment[i] + 1 == alignment[i+1]:
            B += 1
    return B / (len(alignment) + 1)


def make_vocabulary(filepath, vocab_size):
    vocab_counter = {}
    with gzip.open(filepath, 'rb', 'utf-8') as alignfile:
        for _ in alignfile:
            source = alignfile.readline().strip().decode('utf-8')
            line = alignfile.readline().strip().decode('utf-8')
            for word in source.split(' '):
                if word in vocab_counter:
                    vocab_counter[word] += 1
                else:
                    vocab_counter[word] = 1
    vocab_dict = {'<UNK>': 0}
    for w, _ in sorted(vocab_counter.items(), key=lambda x: -x[1]):
        vocab_dict[w] = len(vocab_dict)
        if vocab_size != -1 and len(vocab_dict) >= vocab_size:
            break
    return vocab_dict


def data_prepare(filepath):
    """
    ファイルから入力データと木構造を作る関数
    :param filepath: ファイルパス
    """
    trees = []
    print_message("Reading data...")
    with gzip.open(filepath, 'rb', 'utf-8') as align_file:
        for _ in align_file:
            source = align_file.readline().strip().decode('utf-8')  # 原言語側の文
            target = align_file.readline().strip().decode('utf-8')  # 目的言語側の文
            tree = [Leaf(w, [], idx) for idx, w in enumerate(source.split(' '))]
            target_words_align = re.split('\(\{|\}\)', target)[:-1]
            target_align = [a for i, a in enumerate(target_words_align) if i % 2 == 1]
            for i, a in enumerate(target_align):
                if a.strip():
                    for _a in a.strip().split():
                        tree[int(_a) - 1].alignment.append(i + 1)
            trees.append(tree)
    for tree in trees:
        # yield tree, construct_trees(tree)
        yield tree, cky(tree)


def predict(model, sentence, vocab_dict):
    """
    modelによる木の解析をする関数
    :param model: 解析モデル
    :param sentence: 解析対象
    :return idxes: 構文解析結果（タプルでの木）
    """
    sentence_vectors = [model.leaf(np.array([[vocab_dict[word]]], dtype=xp.int32)) if word in vocab_dict
                        else model.leaf(np.array([[vocab_dict['<UNK>']]], dtype=xp.int32)) for word in sentence]
    idxes = [i for i in range(len(sentence))]
    while len(sentence_vectors) != 1:
        scores = [model.detect_node(left, right).data for left, right in zip(sentence_vectors, sentence_vectors[1:])]
        max_idx = np.argmax(np.array(scores))
        sentence_vectors[max_idx: max_idx + 2] = [model.concat_node(sentence_vectors[max_idx], sentence_vectors[max_idx + 1])]
        pred_label = np.argmax(model.label(sentence_vectors[max_idx]).data)
        if pred_label == 0:
            idxes[max_idx: max_idx + 2] = [(idxes[max_idx], idxes[max_idx + 1])]
        elif pred_label == 1:
            idxes[max_idx: max_idx + 2] = [(idxes[max_idx + 1], idxes[max_idx])]
    return idxes[0]


def traverse(node, model, vocab_dict, node_list, j):
    if isinstance(node, Node):
        left_score, left_loss, left_repr, left_node_list = traverse(node.left, model, vocab_dict, node_list, j)
        right_score, right_loss, right_repr, right_node_list = traverse(node.right, model, vocab_dict, node_list, j)
        if node.label == 'Straight':
            node_repr = model.concat_node(left_repr, right_repr)
            node_score = model.detect_node(left_repr, right_repr)
            label = 0
        else:
            node_repr = model.concat_node(right_repr, left_repr)
            node_score = model.detect_node(right_repr, left_repr)
            label = 1
        if type(left_node_list) == int and type(right_node_list) == int:
            this_node_list = [(left_node_list, right_node_list)]
        elif type(left_node_list) != int and type(right_node_list) == int:
            this_node_list = left_node_list + [(left_node_list[-1], right_node_list)]
        elif type(left_node_list) == int and type(right_node_list) != int:
            this_node_list = right_node_list + [(left_node_list, right_node_list[-1])]
        else:
            this_node_list = left_node_list + right_node_list + [(left_node_list[-1], right_node_list[-1])]
        loss = chainF.softmax_cross_entropy(model.label(node_repr), np.array([label], dtype=np.int32)) \
            + left_loss + right_loss
        node_score += left_score + right_score
        return node_score, loss, node_repr, this_node_list
    else:
        x = vocab_dict[node.word] if node.word in vocab_dict else vocab_dict['<UNK>']
        return 0, 0, model.leaf(xp.array([[x]], dtype=xp.int32)), node.index


def max_gt_score(tree, model, vocab_dict):
    ground_truth_trees = cky(tree)
    max_score, max_score_loss = Variable(xp.array([[0]], dtype=xp.float32)), Variable(xp.array([[0]], dtype=xp.float32))
    node_list = []
    for i, n in enumerate(ground_truth_trees.nodes):
        print('\r%d/%d tree traversed' % (i + 1, len(ground_truth_trees.nodes)), end='')
        score, tree_loss, _, cand_node_list = traverse(n, model, vocab_dict, [], 0)
        if max_score.data < score.data:
            max_score, max_score_loss = score, tree_loss
            node_list = cand_node_list
    print()
    return max_score, max_score_loss, node_list


def max_cand_score(tree, model, vocab_dict, gt_node_list):
    aligns = [l.alignment for l in tree]
    tree_vec = [model.leaf(np.array([[vocab_dict[l.word]]], dtype=xp.int32)) if l.word in vocab_dict
                else model.leaf(np.array([[vocab_dict['<UNK>']]], dtype=xp.int32)) for l in tree]
    concat_nodes = [l for l in range(len(tree_vec))]
    node_list = []
    total_score = 0
    while len(tree_vec) != 1:
        scores = [model.detect_node(left, right) for left, right in zip(tree_vec, tree_vec[1:])]
        max_idx = np.argmax(np.array([score.data for score in scores]))
        total_score += scores[max_idx]
        tree_vec[max_idx: max_idx + 2] = [model.concat_node(tree_vec[max_idx], tree_vec[max_idx + 1])]
        aligns[max_idx: max_idx + 2] = [aligns[max_idx] + aligns[max_idx + 1]]
        node_list.append((concat_nodes[max_idx], concat_nodes[max_idx + 1]))
        concat_nodes[max_idx: max_idx + 2] = [(concat_nodes[max_idx], concat_nodes[max_idx + 1])]
    num_diff = len(set(node_list) - set(gt_node_list))
    return total_score, num_diff


def train(alignment_filepath, epoch, model, optim, bs, vocab_dict):
    trees = []
    # データの読み込み
    print_message('Preparing trees from file...')
    with gzip.open(alignment_filepath, 'rb', 'utf-8') as alignfile:
        for _ in alignfile:
            source = alignfile.readline().strip().decode('utf-8')
            target = alignfile.readline().strip().decode('utf-8')
            tree = [Leaf(w, [], idx) for idx, w in enumerate(source.split(' '))]
            target_words_align = re.split('\(\{|\}\)', target)[:-1]
            target_align = [a for i, a in enumerate(target_words_align) if i % 2 == 1]
            num_align = 0
            for i, a in enumerate(target_align):
                if a.strip():
                    for _a in a.strip().split():
                        num_align += 1
                        tree[int(_a) - 1].alignment.append(i + 1)
            if len(tree) >= 10 or len(tree) >= 2 * num_align:
                continue
            trees.append(tree)
    idxes = [i for i in range(len(trees))]
    for e in range(epoch):
        print_message('Epoch %d start' % (e + 1))
        random.shuffle(idxes)
        total_loss = 0
        batch_loss = 0
        for i, idx in enumerate(idxes):
            if (i + 1) % 10 == 0:
                print_message('%dth tree...' % (i + 1))
            tree = trees[idx]
            ground_truth_score, tree_loss, gt_node_list = max_gt_score(tree, model, vocab_dict)
            cand_score, num_diff = max_cand_score(tree, model, vocab_dict, gt_node_list)
            print('gt score: %.4f' % ground_truth_score.data, 'cand score: %.4f' % cand_score.data, 'diff: %d' % num_diff)
            loss = chainF.squared_error(cand_score + num_diff, ground_truth_score) + tree_loss.reshape((1, 1))
            print('tree loss: %.4f' % loss.data)
            batch_loss += loss
            total_loss += loss.data
            if (i + 1) % bs == 0:
                batch_loss /= bs
                model.cleargrads()
                batch_loss.backward()
                optim.update()
                batch_loss = 0
        # if (i + 1) % bs != 0:
        #     batch_loss /= (i + 1) % bs
        #     model.cleargrads()
        #     batch_loss.backward()
        #     optim.update()
        print_message('loss: %.4f' % total_loss)
    return model


def flatten_tuple_tree(node):
    left, right = node
    if isinstance(left, tuple):
        left_idx = flatten_tuple_tree(left)
    else:
        left_idx = [left]
    if isinstance(right, tuple):
        right_idx = flatten_tuple_tree(right)
    else:
        right_idx = [right]
    return left_idx + right_idx


def main():
    import pprint
    global xp
    args = parse()
    print_message("Prepare training data...")

    model = RecursiveNet(args.vocab_size, args.embed_size, args.hidden_size, args.label)

    if args.gpus >= 0:  # GPU使用の設定
        cuda.get_device_from_id(args.gpus).use()
        model.to_gpu(args.gpus)
        xp = cuda.cupy

    vocab_dict = make_vocabulary(args.alignmentfile, args.vocab_size)

    optm = optimizers.Adam()
    optm.setup(model)
    optm.add_hook(chainer.optimizer.WeightDecay(0.0001))
    optm.add_hook(chainer.optimizer.GradientClipping(5))

    # for c_t in data_prepare(args.alignmentfile):
    #     pprint.pprint(c_t)
    #     input()

    model = train(args.alignmentfile, args.epoch, model, optm, args.batchsize, vocab_dict)
    with gzip.open(args.alignmentfile, 'rb', 'utf-8') as f:
        for _ in f:
            s = f.readline().strip().decode('utf-8').split(' ')
            target = f.readline().strip().decode('utf-8')
            target_word_align = re.split('\(\{|\}\)', target)[:-1]
            pred_idxes = predict(model, s, vocab_dict)
            print(pred_idxes)
            pred_idxes = flatten_tuple_tree(pred_idxes)
            s_idxes = []
            for a in target_word_align[1::2]:
                for _a in a.strip().split():
                    s_idxes.append(pred_idxes.index(int(_a) - 1))
            print("kendall's tau: %.4f" % kendall_tau(s_idxes))
            print(' '.join(s[pred_idx] for pred_idx in pred_idxes))


if __name__ == '__main__':
    main()

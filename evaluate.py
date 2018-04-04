#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse
import codecs
import collections
import gzip
import numpy as np
import pickle
import re

from chainer import cuda, Variable, serializers
import chainer.functions as F

import Recursive_util_pos as util
from Recursive_model import RecursiveNet_CatPos as ReNet


def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('model', help="the model of recursive neural network")
    parser.add_argument('tree_file', help="path of the tree file")
    parser.add_argument('align_file', help="path of the alignment file")
    parser.add_argument('vocab_dict', help="vocabulary dictionary")
    parser.add_argument('--tree_type', default="enju", type=str, help="tree type (enju or s)")
    parser.add_argument('--unit', '-u', default=200, type=int, help="the number of units")
    parser.add_argument('--embed_size', '-emb', default=200, type=int, help="the number of embedding size")
    parser.add_argument('--pos_embed_size', '-pemb', default=200, type=int, help="the number of pos-tag embedding size")
    parser.add_argument('--label', '-l', default=2, type=int, help="the number of labels")

    return parser.parse_args()


def traverse(model, node, train=True, pred=False, root=True, evaluate=None):
    if root:  # 根っこ
        sum_loss = Variable(np.array(0, dtype=np.float32))
        pred_lists = []
        for child in node['children']:
            loss, _, pred_list = traverse(
                model, child, train=train, pred=pred, root=False, evaluate=evaluate
            )
            sum_loss += loss
            if pred_list is not None:
                pred_lists.extend(pred_list)
        return sum_loss, pred_lists
    elif node['tag'] == 'tok':  # 葉ノード
        pred_list = [node['text']]
        embed = np.array([node['node']], dtype=np.int32)
        pos_embed = np.array([node['cat_id']], dtype=np.int32)
        x = Variable(embed)
        p = Variable(pos_embed)
        v = model.leaf(x, p)
        return Variable(np.array(0, dtype=np.float32)), v, pred_list
    else:  # 節ノード
        pred_list = None
        tail = [node['tail']] if node['tail'] else []
        left_node, right_node = node['node']
        left_loss, left, left_pred = traverse(
            model, left_node, train=train, pred=pred, root=False, evaluate=evaluate)
        right_loss, right, right_pred = traverse(
            model, right_node, train=train, pred=pred, root=False, evaluate=evaluate)
        p = Variable(np.array([node['cat_id']], dtype=np.int32))
        v = model.node(left, right, p)
        loss = left_loss + right_loss
        y = model.label(v)
        pred_label = cuda.to_cpu(y.data.argmax(1))
        if train:
            label = np.array([node['label']], dtype=np.int32)
            t = Variable(label)
            loss += F.softmax_cross_entropy(y, t)

        if pred:
            if pred_label[0] == 0:
                left_pred.extend(right_pred)
                pred_list = left_pred + tail
            else:
                right_pred.extend(left_pred)
                pred_list = right_pred + tail

        if evaluate is not None:
            evaluate[node['label']][pred_label[0]] += 1

        return loss, v, pred_list


def read_tree_file(tfile, align_file, v, cv, tt):
    trees = []
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
                tree = util.EnjuXmlParser(line)
            elif tt == "s":
                tree = util.STreeParser(line)
            tree = tree.parse(tree.root)
            if tree['status'] == 'failed':
                continue
            trees.append(convert_tree(tree, v, cv, e_wordlist, f_word_dst, 0, kendall_tau(align)))
    return trees


def convert_tree(node, v, cv, elist, fdst_list, j, tau):
    if node['tag'] == 'sentence':
        children = []
        span = (j, j)
        for child in node['children']:
            span, child_node = convert_tree(child, v, cv, elist, fdst_list, span[1], tau)
            children.append(child_node)
        return {'tag': 'sentence', 'children': children}
    elif node['tag'] == 'cons':
        assert len(node['children']) == 1 or len(node['children']) == 2
        if len(node['children']) == 1:
            return convert_tree(node['children'][0], v, cv, elist, fdst_list, j, tau)
        else:
            swap = 0
            left_span, left_node = convert_tree(node['children'][0], v, cv, elist, fdst_list, j, tau)
            right_span, right_node = convert_tree(node['children'][1], v, cv, elist, fdst_list, max(left_span), tau)
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
                swap = 1
                span_list = right_span + left_span
            else:
                span_list = left_span + right_span
            cat_id = cv[node['cat']] if node['cat'] in cv else cv['<UNK>']
            text = node['text'] if 'text' in node else ""
            tail = node['tail'] if 'tail' in node else ""
            return span_list, {'tag': node['tag'], 'label': swap, 'node': (left_node, right_node), 'cat': node['cat'], 'cat_id': cat_id, 'text': text, 'tail': tail}
    elif node['tag'] == 'tok':
        vob = v[node['text'].lower()] if node['text'].lower() in v else v['<UNK>']
        cat_id = cv[node['pos']] if node['pos'] in cv else cv['<UNK>']
        return [j+1], {'tag': node['tag'], 'node': vob, 'cat_id': cat_id, 'text': node['text'], 'pos': node['pos']}


def kendall_tau(align):
    """
    ケンダールのτの計算、順位相関係数となるようにしたもの
    :param align: アライメント先を表すリスト
    """
    if len(align) == 1:
        return 0.0
    inc = 0
    for i in range(len(align) - 1):
        for j in range(i+1, len(align)):
            if align[i] <= align[j]:
                inc += 1
    try:
        return 2 * inc / (len(align) * (len(align)-1) / 2) -1
    except ZeroDivisionError as _:
        return 0.0


def evaluate(model, eval_trees, result):
    m = model.copy()
    sum_loss = 0
    for tree in eval_trees:
        loss, _ = traverse(m, tree, train=True, pred=False, evaluate=result)
        sum_loss += loss.data
    print_confusion_matrix(result)


def print_confusion_matrix(result):
    print("      |     0|     1")
    print("--------------------")
    print("     0|%6d|%6d %.4f%%" % (result[0][0], result[0][1], result[0][0] / (result[0][0] + result[0][1])))
    print("--------------------")
    print("     1|%6d|%6d %.4f%%" % (result[1][0], result[1][1], result[1][1] / (result[1][0] + result[1][1])))
    print()
    print("acc: %.4f" % ((result[0][0] + result[1][1]) / (result[0][0] + result[0][1] + result[1][0] + result[1][1])))


if __name__ == '__main__':
    args = parse()
    vocab = pickle.load(open(args.vocab_dict, 'rb'))
    cat_vocab = pickle.load(open(args.vocab_dict + '.pos', 'rb'))
    print(len(vocab), len(cat_vocab), args.embed_size, args.pos_embed_size, args.unit, args.label)
    model = ReNet(len(vocab), len(cat_vocab), args.embed_size, args.pos_embed_size, args.unit, args.label)
    serializers.load_hdf5(args.model, model)
    print('Evaluate start...')
    trees = read_tree_file(args.tree_file, args.align_file, vocab, cat_vocab, args.tree_type)
    conf_matrix = [[0 for _ in range(args.label)] for _ in range(args.label)]
    evaluate(model, trees, conf_matrix)

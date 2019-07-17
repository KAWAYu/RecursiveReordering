#!/usr/bin/python3
# encoding: utf-8

import argparse
import codecs
import pickle
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import chainer
from chainer import cuda, optimizers, Variable, serializers
import chainer.functions as F


# xp = numpy or cuda.cupy
xp = np


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
            if node.tail is not None and len(node.tail.strip().split(' ')) == 2:
                this_node['tail'] = node.tail.strip().split(' ')[1].rsplit('/', 1)[0]
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


def parse():
    parser = argparse.ArgumentParser(
        description='Reordering with Recursive Neural Network',
        usage='\n %(prog)s {train, test} filepath alignmentfile [options] reorderfile'
              '\n %(prog)s -h'
    )

    parser.add_argument('mode', help='choose naive or postag')
    parser.add_argument('model', help='model file')
    parser.add_argument('vocab', help='vocabulary dictionary')
    parser.add_argument('reorderfile', nargs='+',
                        help='file path you want to reorder')
    parser.add_argument('--output_format', '-outf', default='text',
                        help='output format (`test` or `order`)')
    parser.add_argument('--unit', '-u', default=30, type=int,
                        help='number of units')
    parser.add_argument('--embed_size', '-emb', default=30, type=int,
                        help='number of embedding size')
    parser.add_argument('--pos_embed_size', '-pemb', default=30, type=int,
                        help='number of pos-tag embedding size')
    parser.add_argument('--label', '-l', default=2, type=int,
                        help='number of labels')
    parser.add_argument('--gpu', '-g', default=-1, type=int,
                        help='number of gpu you want to use')
    parser.add_argument('--tree_type', default='enju', type=str,
                        help="tree type (enju or s)")
    args = parser.parse_args()
    return args


def traverse_pos(model, node, o, train=True, pred=False, root=True, evaluate=None):
    global xp
    if root:  # 根っこ
        sum_loss = Variable(xp.array(0, dtype=np.float32))
        pred_lists, order_lists = [], []
        for child in node['children']:
            loss, _, pred_list, order_list = traverse_pos(
                model, child, o, train=train, pred=pred, root=False, evaluate=evaluate
            )
            sum_loss += loss
            if pred_list is not None:
                pred_lists.extend(pred_list)
            if order_list is not None:
                order_lists.extend(order_list)
                o = len(order_lists)
        return sum_loss, pred_lists, order_lists
    elif node['tag'] == 'tok':  # 葉ノード
        pred_list = [node['text']]
        order_list = [o]
        embed = xp.array([node['node']], dtype=xp.int32)
        pos_embed = xp.array([node['cat_id']], dtype=xp.int32)
        x = Variable(embed)
        p = Variable(pos_embed)
        v = model.leaf(x, p)
        return Variable(xp.array(0, dtype=xp.float32)), v, pred_list, order_list
    else:  # 節ノード
        pred_list = None
        tail, tailo = [], []
        left_node, right_node = node['node']
        left_loss, left, left_pred, left_order = traverse_pos(
            model, left_node, o, train=train, pred=pred, root=False, evaluate=evaluate)
        right_loss, right, right_pred, right_order = traverse_pos(
            model, right_node, o + len(left_order), train=train, pred=pred, root=False, evaluate=evaluate)
        if node['tail']:
            tail = [node['tail']]
            tailo = [o + len(left_order) + len(right_order)]
        p = Variable(xp.array([node['cat_id']], dtype=xp.int32))
        v = model.node(left, right, p)
        loss = left_loss + right_loss
        y = model.label(v)
        pred_label = cuda.to_cpu(y.data.argmax(1))
        if train:
            label = xp.array([node['label']], dtype=np.int32)
            t = chainer.Variable(label)
            loss += F.softmax_cross_entropy(y, t)

        if pred:
            if pred_label[0] == 0:
                left_pred.extend(right_pred)
                pred_list = left_pred + tail
                left_order.extend(right_order)
                order_list = left_order + tailo
            else:
                right_pred.extend(left_pred)
                pred_list = right_pred + tail
                right_order.extend(left_order)
                order_list = right_order + tailo

        if evaluate is not None:
            if pred_label[0] == node['label']:
                evaluate['correct_node'] += 1
            evaluate['total_node'] += 1
        return loss, v, pred_list, order_list


def traverse(model, node, o, train=True, pred=False, root=True, evaluate=None):
    global xp
    if root:  # 根っこ
        sum_loss = Variable(xp.array(0, dtype=np.float32))
        pred_lists = []
        for child in node['children']:
            loss, _, pred_list = traverse_pos(
                model, child, train=train, pred=pred, root=False, evaluate=evaluate
            )
            sum_loss += loss
            if pred_list is not None:
                pred_lists.extend(pred_list)
        return sum_loss, pred_lists
    elif node['tag'] == 'tok':  # 葉ノード
        pred_list = [node['text']]
        embed = xp.array([node['node']], dtype=xp.int32)
        x = Variable(embed)
        v = model.leaf(x)
        return Variable(xp.array(0, dtype=xp.float32)), v, pred_list
    else:  # 節ノード
        pred_list = None
        tail = [node['tail']] if node['tail'] else []
        left_node, right_node = node['node']
        left_loss, left, left_pred = traverse_pos(
            model, left_node, train=train, pred=pred, root=False, evaluate=evaluate)
        right_loss, right, right_pred = traverse_pos(
            model, right_node, train=train, pred=pred, root=False, evaluate=evaluate)
        v = model.node(left, right)
        loss = left_loss + right_loss
        y = model.label(v)
        pred_label = cuda.to_cpu(y.data.argmax(1))
        if train:
            label = xp.array([node['label']], dtype=np.int32)
            t = chainer.Variable(label)
            loss += F.softmax_cross_entropy(y, t)

        if pred:
            if pred_label[0] == 0:
                left_pred.extend(right_pred)
                pred_list = left_pred + tail
            else:
                right_pred.extend(left_pred)
                pred_list = right_pred + tail

        if evaluate is not None:
            if pred_label[0] == node['label']:
                evaluate['correct_node'] += 1
            evaluate['total_node'] += 1
        return loss, v, pred_list


def read_reorder_pos(tree_file_path, vocab, cat_vocab, tree_type):
    with codecs.open(tree_file_path, 'r', 'utf-8') as tree_file:
        for line in tree_file:
            line = line.strip()
            if tree_type == "enju":
                tree = EnjuXmlParser(line)
            elif tree_type == "s":
                tree = STreeParser(line)
            tree = tree.parse(tree.root)
            if tree['status'] == 'failed':
                continue
            yield convert_tree_reorder_pos(vocab, tree, cat_vocab)


def read_reorder(tree_file_path, vocab, tree_type):
    with codecs.open(tree_file_path, 'r', 'utf-8') as tree_file:
        for line in tree_file:
            line = line.strip()
            if tree_type == 'enju':
                tree = EnjuXmlParser(line)
            elif tree_type == 's':
                tree = STreeParser(line)
            tree = tree.parse(tree.root)
            if tree['status'] == 'failed':
                continue
            yield convert_tree_reorder(vocab, tree)


def convert_tree_reorder_pos(vocab, node, cat_vocab):
    """
    並び替えたいファイルのデータを読み込む（テストデータ用）
    :param vocab: 単語->idの辞書
    :param node: 今のノード
    """
    if node['tag'] == 'sentence':
        children = []
        for child in node['children']:
            children.append(convert_tree_reorder_pos(vocab, child, cat_vocab))
        return {'tag': 'sentence', 'children': children}
    elif node['tag'] == 'cons':
        assert len(node['children']) == 1 or len(node['children']) == 2
        if len(node['children']) == 1:
            return convert_tree_reorder_pos(vocab, node['children'][0], cat_vocab)
        else:
            left_node = convert_tree_reorder_pos(vocab, node['children'][0], cat_vocab)
            right_node = convert_tree_reorder_pos(vocab, node['children'][1], cat_vocab)
            cat_id = cat_vocab[node['cat']] if node['cat'] in cat_vocab else cat_vocab['<UNK>']
            text = node['text'] if 'text' in node else ""
            tail = node['tail'] if 'tail' in node else ""
            return {'tag': node['tag'], 'node': (left_node, right_node), 'cat': node['cat'], 'cat_id': cat_id, 'text': text, 'tail': tail}
    elif node['tag'] == 'tok':
        v = vocab[node['text'].lower()] if node['text'].lower() in vocab else vocab['<UNK>']
        cat_id = cat_vocab[node['pos']] if node['pos'] in cat_vocab else cat_vocab['<UNK>']
        return {'tag': node['tag'], 'node': v, 'cat_id': cat_id, 'text': node['text'], 'pos': node['pos']}


def convert_tree_reorder(vocab, node):
    """
    並び替えたいファイルのデータを読み込む（テストデータ用）
    :param vocab: 単語->idの辞書
    :param node: 今のノード
    """
    if node['tag'] == 'sentence':
        children = []
        for child in node['children']:
            children.append(convert_tree_reorder(vocab, child))
        return {'tag': 'sentence', 'children': children}
    elif node['tag'] == 'cons':
        assert len(node['children']) == 1 or len(node['children']) == 2
        if len(node['children']) == 1:
            return convert_tree_reorder(vocab, node['children'][0])
        else:
            left_node = convert_tree_reorder(vocab, node['children'][0])
            right_node = convert_tree_reorder(vocab, node['children'][1])
            text = node['text'] if 'text' in node else ''
            tail = node['tail'] if 'tail' in node else ''
            return {'tag': node['tag'], 'node': (left_node, right_node), 'cat': node['cat'], 'text': text, 'tail': tail}
    elif node['tag'] == 'tok':
        v = vocab[node['text'].lower()] if node['text'].lower() in vocab else vocab['<UNK>']
        return {'tag': node['tag'], 'node': v, 'text': node['text'], 'pos_tag': node['pos']}


if __name__ == '__main__':
    args = parse()
    vocab = pickle.load(open(args.vocab, 'rb'))

    print("Loading Model...")
    if args.mode == 'naive':
        from Recursive_model import RecursiveNet
        model = RecursiveNet(len(vocab), args.embed_size, args.unit, args.label)
        if args.gpu >= 0:
            cuda.get_device(args.gpus).use()
            model.to_gpu(args.gpu)
            xp = cuda.cupy
        serializers.load_hdf5(args.model, model)
        print("Begin reordering...")
        for i, fp in enumerate(args.reorderfile):
            with codecs.open(fp.split('/')[-1] + '.re', 'w', 'utf-8') as fre:
                for tree in read_reorder(fp, vocab, args.tree_type):
                    _, pred, order = traverse(model, tree, 0, train=False, pred=True)
                    if args.output_format == 'text':
                        print(' '.join(pred), file=fre)
                    elif args.output_format == 'order':
                        hd = {}
                        for oi, o in enumerate(order):
                            hd[o] = oi
                        print(' '.join(str(hd[oi]) for oi in range(len(order))), file=fre)
    elif args.mode == 'postag':
        from Recursive_model import RecursiveNet_CatPos
        cat_vocab = pickle.load(open(args.vocab + '.pos', 'rb'))
        model = RecursiveNet_CatPos(len(vocab), len(cat_vocab), args.embed_size, args.pos_embed_size, args.unit, args.label)
        if args.gpu >= 0:  # GPU使用の設定
            cuda.get_device(args.gpu).use()
            model.to_gpu(args.gpu)
            xp = cuda.cupy
        serializers.load_hdf5(args.model, model)
        print("Begin reordering...")
        for i, fp in enumerate(args.reorderfile):
            with codecs.open(fp.split('/')[-1]+'.re', 'w', 'utf-8') as fre:
                for tree in read_reorder_pos(fp, vocab, cat_vocab, args.tree_type):
                    _, pred, order = traverse_pos(model, tree, 0, train=False, pred=True)
                    if args.output_format == 'text':
                        print(' '.join(pred), file=fre)
                    elif args.output_format == 'order':
                        hd = {}
                        for oi, o in enumerate(order):
                            hd[o] = oi
                        print(' '.join(str(hd[oi]) for oi in range(len(order))), file=fre)

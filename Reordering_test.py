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
            if node.tail and len(node.tail.split()) == 2:
                this_node['tail'] = node.tail.split()[1].rsplit('/')[0]
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
        description = 'Reordering with Recursive Neural Network',
        usage = '\n %(prog)s {train, test} filepath alignmentfile [options] reorderfile'
            '\n %(prog)s -h'
    )

    parser.add_argument('mode', help='choose naive or postag')
    parser.add_argument('model', help='model file')
    parser.add_argument('vocab', help='vocabulary dictionary')
    parser.add_argument('reorderfile', nargs='+',
        help='file path you want to reorder')
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
    args = parser.parse_args()
    return args


def traverse(model, node, train=True, pred=False, root=True, evaluate=None):
    global xp
    if root:  # 根っこ
        sum_loss = Variable(xp.array(0, dtype=np.float32))
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
        embed = xp.array([node['node']], dtype=xp.int32)
        pos_embed = xp.array([node['cat_id']], dtype=xp.int32)
        x = Variable(embed)
        p = Variable(pos_embed)
        v = model.leaf(x, p)
        return Variable(xp.array(0, dtype=xp.float32)), v, pred_list
    else:  # 節ノード
        pred_list = None
        tail = [node['tail']] if node['tail'] else []
        left_node, right_node = node['node']
        left_loss, left, left_pred = traverse(
            model, left_node, train=train, pred=pred, root=False, evaluate=evaluate)
        right_loss, right, right_pred = traverse(
            model, right_node, train=train, pred=pred, root=False, evaluate=evaluate)
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
            else:
                right_pred.extend(left_pred)
                pred_list = right_pred + tail

        if evaluate is not None:
            if pred_label[0] == node['label']:
                evaluate['correct_node'] += 1
            evaluate['total_node'] += 1
        return loss, v, pred_list


def read_reorder_pos(tree_file_path, vocab, cat_vocab):
    with codecs.open(tree_file_path, 'r', 'utf-8') as tree_file:
        for line in tree_file:
            line = line.strip()
            tree = EnjuXmlParser(line)
            tree = tree.parse(tree.root)
            if tree['status'] == 'failed':
                continue
            yield convert_tree_reorder_pos(vocab, tree, cat_vocab)


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


def read_reorder(tree_file_path, vocab):
    with codecs.open(tree_file_path, 'r', 'utf-8') as tree_file:
        for line in tree_file:
            line = line.strip()
            tree = EnjuXmlParser(line)
            tree = tree.parse(tree.root)
            if tree['status'] == 'failed':
                continue
            yield convert_tree_reorder(vocab, tree)


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
            return {'node': (left_node, right_node), 'cat': node['cat']}
    elif node['tag'] == 'tok':
        v = vocab[node['text'].lower()] if node['text'].lower() in vocab else vocab['<UNK>']
        return {'node': v, 'text': node['text'], 'pos_tag': node['pos']}


if __name__ == '__main__':
    args = parse()
    vocab = pickle.load(open(args.vocab, 'rb'))

    print("Loading Model...")
    if args.mode == 'naive':
        from Recursive_model import RecursiveNet
        import Recursive_util as util
        #rtrees = [util.read_reorder(fp, vocab, None) for fp in args.reorderfile]
        model = RecursiveNet(len(vocab), args.embed_size, args.unit, args.label)
        serializers.load_hdf5(args.model, model)
        print("Begin reordering...")
        for i, fp in enumerate(args.reorderfile):
            with codecs.open(fp+'.re', 'w', 'utf-8') as fre:
                for tree in read_reorder(fp, vocab):
                    _, pred = traverse(model, tree, train=False, pred=True)
                    print(' '.join(pred), file=fre)
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
                for tree in read_reorder_pos(fp, vocab, cat_vocab):
                    _, pred = traverse(model, tree, train=False, pred=True)
                    print(' '.join(pred), file=fre)


# -*- coding: utf-8 -*-

import xml.etree.ElementTree as ET
import codecs
import os
import re
import sys
import gzip
import collections

from my_utils.cc_rank import kendall_tau


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


def read_reorder(tree_file_path, vocab, tree_reorder):
    """
    並び替えたいファイルを読み込むための関数。並び替えるだけなのでアライメント無し。
    """
    trees = []
    with codecs.open(tree_file_path, 'r', 'utf-8') as tree_file:
        for line in tree_file:
            line = line.strip()
            if tree_reorder == 'enju':
                tree = EnjuXmlParser(line)
            elif tree_reorder == 's':
                tree = STreeParser(line)
            else:
                print('invalid tree parser', file=sys.stderr)
                sys.exit(1)
            tree = tree.parse(tree.root)
            if tree['status'] == 'failed':
                continue
            trees.append(convert_tree_reorder(vocab, tree))
    return trees


def read_dev(tree_file_path, align_file_path, vocab, tree_parser):
    """
    検証ファイルを読み込むための関数
    """
    trees = []
    with codecs.open(tree_file_path, 'r', 'utf-8') as tree_file_path, gzip.open(align_file_path, 'r') as align_file:
        for i, tree_line in enumerate(tree_file_path):
            if (i + 1) % 1000 == 0:
                print("%d lines have been read..." % (i + 1))
            tree_line = tree_line.strip()
            align_file.readline()  # 3n+1行目は不要なので飛ばす
            # 原言語の読み込み(3n+2行目)
            e_words = align_file.readline().strip().decode('utf-8').split()
            # 目的言語の読み込み(3n行目)
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
            for j in range(len(f_words) // 2):
                # NULLアライメントは考慮しないのでfor文は1から
                f_word = f_words[2 * j]  # 目的言語側の単語
                f_align = f_words[2 * j + 1].strip().split()  # 目的言語のアライメント先
                f_word_dst.append((f_word, [e_wordlist[int(k) - 1] for k in f_align]))
                align.extend([int(k) for k in f_align])

            if tree_parser == 'enju':
                tree = EnjuXmlParser(tree_line)
            elif tree_parser == 's':
                tree = STreeParser(tree_line)
            else:
                print('invalid tree parser', file=sys.stderr)
                sys.exit(1)
            tree = tree.parse(tree.root)
            if tree['status'] == 'failed':
                continue

            trees.append(convert_tree_dev(vocab, tree, e_wordlist, f_word_dst, 0, kendall_tau(align)))
    return trees


def read_train(tree_file_path, align_file_path, vocab, max_size, vocab_size, tree_parser):
    """
    訓練用のファイルを読み込むための関数。
    """
    trees = []
    with codecs.open(tree_file_path, 'r', 'utf-8') as tree_file_path, gzip.open(align_file_path, 'r') as align_file:
        for i, tree_line in enumerate(tree_file_path):
            if (i + 1) % 1000 == 0:
                print("%d lines have been read..." % (i + 1))
            tree_line = tree_line.strip()
            align_file.readline()  # 3n+1行目は不要なので飛ばす
            # 原言語の読み込み(3n+2行目)
            e_words = align_file.readline().strip().decode('utf-8').split()
            # 目的言語の読み込み(3n行目)
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
            # for j in range(1, len(f_words)//2):
            # NULLアライメントは考慮しないのでfor文は1から
            for j in range(len(f_words) // 2):
                f_word = f_words[2 * j]  # 目的言語側の単語
                f_align = f_words[2 * j + 1].strip().split()  # 目的言語のアライメント先
                f_word_dst.append((f_word, [e_wordlist[int(k) - 1] for k in f_align]))
                align.extend([int(k) for k in f_align])

            if tree_parser == 'enju':
                tree = EnjuXmlParser(tree_line)
            elif tree_parser == 's':
                tree = STreeParser(tree_line)
            else:
                print('invalid tree parser', file=sys.stderr)
                sys.exit(1)
            tree = tree.parse(tree.root)
            if tree['status'] == 'failed':
                continue
            trees.append(
                convert_tree_train(vocab, tree, e_wordlist, f_word_dst, 0, kendall_tau(align), vocab_size))
            if max_size and len(trees) >= max_size:
                break
    return trees, vocab


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
        return {'tag': node['tag'], 'children': children}
    elif node['tag'] == 'cons':
        assert len(node['children']) == 1 or len(node['children']) == 2
        if len(node['children']) == 1:
            return convert_tree_reorder(vocab, node['children'][0])
        else:
            left_node = convert_tree_reorder(vocab, node['children'][0])
            right_node = convert_tree_reorder(vocab, node['children'][1])
            text = node['text'] if 'text' in node else ""
            tail = node['tail'] if 'tail' in node else ""
            return {'tag': node['tag'], 'node': (left_node, right_node), 'cat': node['cat'],
                    'text': text, 'tail': tail}
    elif node['tag'] == 'tok':
        v = vocab[node['text'].lower()] if node['text'].lower() in vocab else vocab['<UNK>']
        return {'tag': node['tag'], 'node': v, 'text': node['text'], 'pos': node['pos']}


def convert_tree_train(vocab, node, e_list, f_dst_list, j, tau, v_size):
    """
    :param vocab: 単語->idの辞書
    :param node: 今のノード
    :param e_list: 原言語のリスト
    :param f_dst_list: 目的言語がどの原言語に対応しているかのリスト
    :param j: 被覆している単語の右端が？？？
    :param tau: 今のケンダールのτ
    :param v_size: 単語の辞書の大きさ
    """
    if node['tag'] == 'sentence':
        children = []
        span = (j, j)
        for child in node['children']:
            span, child_node = convert_tree_train(vocab, child, e_list, f_dst_list, span[1], tau, v_size)
            children.append(child_node)
        return {'tag': node['tag'], 'children': children}
    elif node['tag'] == 'cons':
        assert len(node['children']) == 1 or len(node['children']) == 2
        if len(node['children']) == 1:
            return convert_tree_train(vocab, node['children'][0], e_list, f_dst_list, j, tau, v_size)
        else:
            swap = 0
            left_span, left_node = convert_tree_train(vocab, node['children'][0], e_list, f_dst_list, j, tau, v_size)
            right_span, right_node = convert_tree_train(vocab, node['children'][1], e_list, f_dst_list, max(left_span),
                                                        tau, v_size)
            # 並び替え候補の作成
            span_min, span_max = min(left_span), max(right_span)
            # 並び替えなしの時の単語の順番
            tmp_e_list = [e_list[i] for i in range(span_min - 1)] + [e_list[i - 1] for i in left_span] \
                         + [e_list[i - 1] for i in right_span] + [e_list[i] for i in range(span_max, len(e_list))]
            align = []
            for f_w_dst in f_dst_list:
                for e_dst in f_w_dst[1]:
                    if e_dst in tmp_e_list:
                        align.append(tmp_e_list.index(e_dst))
            tmp_tau_1 = kendall_tau(align)
            # 並び替えあり
            tmp_e_list = [e_list[i] for i in range(span_min - 1)]
            tmp_e_list.extend([e_list[i - 1] for i in right_span])
            tmp_e_list.extend([e_list[i - 1] for i in left_span])
            tmp_e_list.extend([e_list[i] for i in range(span_max, len(e_list))])
            align = []
            for f_w_dst in f_dst_list:
                for e_dst in f_w_dst[1]:
                    if e_dst in tmp_e_list:
                        align.append(tmp_e_list.index(e_dst))
            tmp_tau_2 = kendall_tau(align)
            if tmp_tau_2 > tmp_tau_1:
                swap = 1
                span_list = right_span + left_span
            else:
                span_list = left_span + right_span
            text = node['text'] if 'text' in node else ""
            tail = node['tail'] if 'tail' in node else ""
            return span_list, {'tag': node['tag'], 'label': swap, 'node': (left_node, right_node), 'cat': node['cat'],
                               'text': text, 'tail': tail}
    elif node['tag'] == 'tok':
        if node['text'].lower() not in vocab and (v_size == -1 or len(vocab) < v_size):
            vocab[node['text'].lower()] = len(vocab)
        t = vocab[node['text'].lower()] if node['text'].lower() in vocab else vocab['<UNK>']
        return [j + 1], {'tag': node['tag'], 'node': t, 'text': node['text'], 'pos': node['pos']}


def convert_tree_dev(vocab, node, e_list, f_dst_list, j, tau):
    """
    :param e_list: 原言語のリスト
    :param f_dst_list: 目的言語がどの原言語に対応しているかのリスト
    :param j: 被覆している単語の右端が？？？
    :param tau: 今のケンダールのτ
    """
    if node['tag'] == 'sentence':
        children = []
        span = (j, j)
        for child in node['children']:
            span, child_node = convert_tree_dev(vocab, child, e_list, f_dst_list, span[1], tau)
            children.append(child_node)
        return {'tag': node['tag'], 'children': children}
    elif node['tag'] == 'cons':
        assert len(node['children']) == 1 or len(node['children']) == 2
        if len(node['children']) == 1:
            return convert_tree_dev(vocab, node['children'][0], e_list, f_dst_list, j, tau)
        else:
            swap = 0
            left_span, left_node = convert_tree_dev(vocab, node['children'][0], e_list, f_dst_list, j, tau)
            right_span, right_node = convert_tree_dev(vocab, node['children'][1], e_list, f_dst_list, max(left_span),
                                                      tau)
            # 並び替え候補の作成
            span_min, span_max = min(left_span), max(right_span)
            # 並び替えなしの時の単語の順番
            tmp_e_list = [e_list[i] for i in range(span_min - 1)] + [e_list[i - 1] for i in left_span] \
                         + [e_list[i - 1] for i in right_span] + [e_list[i] for i in range(span_max, len(e_list))]
            align = []
            for f_w_dst in f_dst_list:
                for e_dst in f_w_dst[1]:
                    if e_dst in tmp_e_list:
                        align.append(tmp_e_list.index(e_dst))
            tmp_tau_1 = kendall_tau(align)
            # 並び替えあり
            tmp_e_list = [e_list[i] for i in range(span_min - 1)]
            tmp_e_list.extend([e_list[i - 1] for i in right_span])
            tmp_e_list.extend([e_list[i - 1] for i in left_span])
            tmp_e_list.extend([e_list[i] for i in range(span_max, len(e_list))])
            align = []
            for f_w_dst in f_dst_list:
                for e_dst in f_w_dst[1]:
                    if e_dst in tmp_e_list:
                        align.append(tmp_e_list.index(e_dst))
            tmp_tau_2 = kendall_tau(align)
            if tmp_tau_2 > tmp_tau_1:
                swap = 1
                span_list = right_span + left_span
            else:
                span_list = left_span + right_span
            text = node['text'] if 'text' in node else ""
            tail = node['tail'] if 'tail' in node else ""
            return span_list, {'tag': node['tag'], 'label': swap, 'node': (left_node, right_node), 'cat': node['cat'],
                               'text': text, 'tail': tail}
    elif node['tag'] == 'tok':
        t = vocab[node['text'].lower()] if node['text'].lower() in vocab else vocab['<UNK>']
        return [j + 1], {'tag': node['tag'], 'node': t, 'text': node['text'], 'pos': node['pos']}


if __name__ == "__main__":
    import pprint

    tree_file, align_file = sys.argv[1], sys.argv[2]
    vocab = {}
    trees = read_train(tree_file, align_file, vocab, None)
    pprint.pprint(trees)

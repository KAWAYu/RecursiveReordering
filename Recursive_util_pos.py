# -*- coding: utf-8 -*-

import xml.etree.ElementTree as ET
from io import open
import re
import sys
import gzip
import collections


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
            status = node.attrib['parse_status']
            this_node['status'] = status
        elif tag == 'cons':
            this_node['cat'] = node.attrib['cat']
            this_node['head'] = node.attrib['head']
            if node.text:
                this_node['text'] = node.text
            if node.tail is not None and len(node.tail.strip().split(' ')) >= 2:
                tail = [t.rsplit('/', 1)[0] for t in node.tail.strip().split(' ')[1:]
                        if len(t.rsplit('/', 1)) == 2 and t.rsplit('/', 1)[1] == '.']
                this_node['tail'] = tail
        elif tag == 'tok':
            this_node['cat'] = node.attrib['cat']
            this_node['pos'] = node.attrib['pos']
            this_node['text'] = node.text
        this_node['id'] = node.attrib['id']
        this_node['tag'] = tag
        this_node['children'] = children
        return this_node


def read_reorder(tree_file_path, vocab, cat_vocab, tree_reorder):
    """
    並び替えたいファイルを読み込むための関数。並び替えるだけなのでアライメント無し。
    """
    trees = []
    with open(tree_file_path, 'r', encoding='utf-8') as tree_file:
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
            trees.append(convert_tree_reorder(vocab, tree, cat_vocab))
    return trees


def read_dev(tree_file_path, align_file_path, vocab, cat_vocab, tree_parser):
    """
    検証ファイルを読み込むための関数
    """
    trees = []
    with open(tree_file_path, 'r', encoding='utf-8') as tree_file_path, gzip.open(align_file_path, 'r') as align_file:
        for i, tree_line in enumerate(tree_file_path):
            if (i+1) % 1000 == 0:
                print("%d lines have been read..." % (i+1))
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
            for j in range(len(f_words)//2):
                # NULLアライメントは考慮しないのでfor文は1から
                f_word = f_words[2*j]  # 目的言語側の単語
                f_align = f_words[2*j+1].strip().split()  # 目的言語のアライメント先
                f_word_dst.append((f_word, [e_wordlist[int(k)-1] for k in f_align]))
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
            
            trees.append(convert_tree_dev(vocab, tree, e_wordlist, f_word_dst, 0, kendall_tau(align), cat_vocab))
    return trees


def read_train(tree_file_path, align_file_path, vocab, max_size, vocab_size, cat_vocab, tree_parser):
    """
    訓練用のファイルを読み込むための関数。
    """
    trees = []
    with open(tree_file_path, 'r', encoding='utf-8') as tree_file_path, gzip.open(align_file_path, 'r') as align_file:
        for i, tree_line in enumerate(tree_file_path):
            if (i+1) % 1000 == 0:
                print("%d lines have been read..." % (i+1))
            tree_line = tree_line.strip()
            align_file.readline()  # 3n+1行目は不要なので飛ばす
            # 原言語の読み込み(3n+2行目)
            e_words = align_file.readline().strip().decode('utf-8').split(' ')
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
                f_word = f_words[2*j]  # 目的言語側の単語
                f_align = f_words[2*j+1].strip().split()  # 目的言語のアライメント先
                f_word_dst.append((f_word, [e_wordlist[int(k)-1] for k in f_align]))
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
            trees.append(convert_tree_train(vocab, tree, e_wordlist, f_word_dst, 0, kendall_tau(align), vocab_size, cat_vocab))
            if max_size and len(trees) >= max_size:
                break
    return trees, vocab


def convert_tree_reorder(vocab, node, cat_vocab):
    """
    並び替えたいファイルのデータを読み込む（テストデータ用）
    :param vocab: 単語->idの辞書
    :param node: 今のノード
    """
    if node['tag'] == 'sentence':
        children = []
        for child in node['children']:
            children.append(convert_tree_reorder(vocab, child, cat_vocab))
        return {'tag': node['tag'], 'children': children}
    elif node['tag'] == 'cons':
        assert len(node['children']) == 1 or len(node['children']) == 2
        if len(node['children']) == 1:
            tail = node['tail'] if 'tail' in node else []
            cnode = convert_tree_reorder(vocab, node['children'][0], cat_vocab)
            if 'tail' in cnode:
                cnode['tail'] += tail
            else:
                cnode['tail'] = tail
            return cnode
        else:
            left_node = convert_tree_reorder(vocab, node['children'][0], cat_vocab)
            right_node = convert_tree_reorder(vocab, node['children'][1], cat_vocab)
            cat_id = cat_vocab[node['cat']] if node['cat'] in cat_vocab else cat_vocab['<UNK>']
            text = node['text'] if 'text' in node else ""
            tail = node['tail'] if 'tail' in node else []
            return {'tag': node['tag'], 'node': (left_node, right_node), 'cat': node['cat'], 'cat_id': cat_id, 'text': text, 'tail': tail}
    elif node['tag'] == 'tok':
        v = vocab[node['text'].lower()] if node['text'].lower() in vocab else vocab['<UNK>']
        cat_id = cat_vocab[node['pos']] if node['pos'] in cat_vocab else cat_vocab['<UNK>']
        return {'tag': node['tag'], 'node': v, 'cat_id': cat_id, 'text': node['text'], 'pos': node['pos']}


def convert_tree_train(vocab, node, e_list, f_dst_list, j, tau, v_size, cat_vocab):
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
            span, child_node = convert_tree_train(vocab, child, e_list, f_dst_list, span[1], tau, v_size, cat_vocab)
            children.append(child_node)
        return {'tag': node['tag'], 'children': children}
    elif node['tag'] == 'cons':
        assert len(node['children']) == 1 or len(node['children']) == 2
        if len(node['children']) == 1:
            cspan, cnode = convert_tree_train(vocab, node['children'][0], e_list, f_dst_list, j, tau, v_size, cat_vocab)
            max_span = max(cspan)
            if 'tail' in node:
                tail_span = [max_span+i+1 for i, _ in enumerate(node['tail'])]
            else:
                tail_span = []
            if 'tail' in cnode:
                cnode['tail'] += node['tail'] if 'tail' in node else []
            else:
                cnode['tail'] = node['tail'] if 'tail' in node else []
            return cspan + tail_span, cnode
        else:
            swap = 0
            left_span, left_node = convert_tree_train(vocab, node['children'][0], e_list, f_dst_list, j, tau, v_size, cat_vocab)
            right_span, right_node = convert_tree_train(vocab, node['children'][1], e_list, f_dst_list, max(left_span), tau, v_size, cat_vocab)
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
            tmp_tau_1 = kendall_tau(align)
            # 並び替えあり
            tmp_e_list = [e_list[i] for i in range(span_min - 1)]
            tmp_e_list.extend([e_list[i-1] for i in right_span])
            tmp_e_list.extend([e_list[i-1] for i in left_span])
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
            if node['cat'] not in cat_vocab:
                cat_vocab[node['cat']] = len(cat_vocab)
            cat_id = cat_vocab[node['cat']] if node['cat'] in cat_vocab else cat_vocab['<UNK>']
            text = node['text'] if 'text' in node else ""
            tail = node['tail'] if 'tail' in node else []
            return span_list, {'tag': node['tag'], 'label': swap, 'node': (left_node, right_node), 'cat': node['cat'], 'cat_id': cat_id, 'text': text, 'tail': tail}
    elif node['tag'] == 'tok':
        if node['text'].lower() not in vocab and (v_size == -1 or len(vocab) < v_size):
            vocab[node['text'].lower()] = len(vocab)
        t = vocab[node['text'].lower()] if node['text'].lower() in vocab else vocab['<UNK>']
        if node['pos'] not in cat_vocab:
            cat_vocab[node['pos']] = len(cat_vocab)
        cat_id = cat_vocab[node['pos']]
        return [j+1], {'tag': node['tag'], 'node': t, 'text': node['text'], 'pos': node['pos'], 'cat_id': cat_id}


def convert_tree_dev(vocab, node, e_list, f_dst_list, j, tau, cat_vocab):
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
            span, child_node = convert_tree_dev(vocab, child, e_list, f_dst_list, span[1], tau, cat_vocab)
            children.append(child_node)
        return {'tag': node['tag'], 'children': children}
    elif node['tag'] == 'cons':
        assert len(node['children']) == 1 or len(node['children']) == 2
        if len(node['children']) == 1:
            cspan, cnode = convert_tree_dev(vocab, node['children'][0], e_list, f_dst_list, j, tau, cat_vocab)
            max_span = max(cspan)
            if 'tail' in node:
                tail_span = [max_span + i + 1 for i, _ in enumerate(node['tail'])]
            else:
                tail_span = []
            if 'tail' in cnode:
                cnode['tail'] += node['tail'] if 'tail' in node else []
            else:
                cnode['tail'] = node['tail'] if 'tail' in node else []
            return cspan + tail_span, cnode
        else:
            swap = 0
            left_span, left_node = convert_tree_dev(vocab, node['children'][0], e_list, f_dst_list, j, tau, cat_vocab)
            right_span, right_node = convert_tree_dev(vocab, node['children'][1], e_list, f_dst_list, max(left_span), tau, cat_vocab)
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
            tmp_tau_1 = kendall_tau(align)
            # 並び替えあり
            tmp_e_list = [e_list[i] for i in range(span_min - 1)]
            tmp_e_list.extend([e_list[i-1] for i in right_span])
            tmp_e_list.extend([e_list[i-1] for i in left_span])
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
            cat_id = cat_vocab[node['cat']] if node['cat'] in cat_vocab else cat_vocab['<UNK>']
            text = node['text'] if 'text' in node else ""
            tail = node['tail'] if 'tail' in node else []
            return span_list, {'tag': node['tag'], 'label': swap, 'node': (left_node, right_node), 'cat': node['cat'], 'cat_id': cat_id, 'text': text, 'tail': tail}
    elif node['tag'] == 'tok':
        t = vocab[node['text'].lower()] if node['text'].lower() in vocab else vocab['<UNK>']
        cat_id = cat_vocab[node['pos']] if node['pos'] in cat_vocab else cat_vocab['<UNK>']
        return [j+1], {'tag': node['tag'], 'node': t, 'text': node['text'], 'pos': node['pos'], 'cat_id': cat_id}


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


if __name__ == "__main__":
    import pprint
    tree_file, align_file = sys.argv[1], sys.argv[2]
    vocab = {}
    trees = read_train(tree_file, align_file, vocab, None)
    pprint.pprint(trees)

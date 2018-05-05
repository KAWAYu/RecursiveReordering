#!/usr/bin/python
# -*- coding: utf-8 -*-

import codecs
import sys


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
            return self.parse(node[-1], root=False)
        elif len(node) == 2:  # 子が二つの時
            if isinstance(node[1], str):  # 葉ノード
                return [node[1] + '/' + node[0]]
            else:  # 節ノード
                return self.parse(node[1], root=False)
        else:  # 子が三つの時
            this_node['tag'] = 'cons'
            left_node = self.parse(node[1], root=False)
            right_node = self.parse(node[2], root=False)
            return left_node + right_node


if __name__ == '__main__':
    with codecs.open(sys.argv[1]) as fin, codecs.open(sys.argv[2], 'w') as fout:
        for line in fin:
            tree = STreeParser(line.strip())
            poses = tree.parse(tree.root)
            if 'status' in poses:
                continue
            print(' '.join(poses), file=fout)

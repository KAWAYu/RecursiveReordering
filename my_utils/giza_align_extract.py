# -*- coding: utf-8 -*-

import codecs
import argparse
import sys


class AlignmentSentences(object):
    """
    原言語側の文、目的言語側の文、原言語<->目的言語アライメントを保持するクラス
    
    """
    def __init__(self, src_snt, tgt_snt, s2t_align, t2s_align):
        self.src_snt = src_snt  # 原言語側の文
        self.tgt_snt = tgt_snt  # 目的言語側の文
        self.s2t_align = self._process_align_line(s2t_align)  # src->tgtのアライメント
        self.t2s_align = self._process_align_line(t2s_align)  # tgt->srcのアライメント

    def _process_align_line(self, line):
        s_ids = [int(s_id) for i, s_id in enumerate(line.split()) if i % 2 == 0]
        t_ids = [int(t_id) for i, t_id in enumerate(line.split()) if i % 2 == 1]
        return [(s_id, t_id) for s_id, t_id in zip(s_ids, t_ids)]

    def _extract_union(self):
        return [(s_id, t_id) for s_id, t_id in self.s2t_align if s_id == t2s_ids[1] or t_id == t2s_ids[0]
                for t2s_ids in self.t2s_align]

    def _extract_intersection(self):
        return [(s_id, t_id) for s_id, t_id in self.s2t_align for t2s_ids in self.t2s_align
                if s_id == t2s_ids[1] and t_id == t2s_ids[0]]

    def extract_align(self, method='grow'):
        if method == 'intersection':
            return self._extract_intersection()
        elif method == 'union':
            return self._extract_union()
        elif method == 'grow':
            raise NotImplementedError('grow: Sorry for incovinience')
        elif method == 'grow-diag':
            raise NotImplementedError('grow-diag: Sorry for incovinience')
        elif method == 'grow-diag-final':
            raise NotImplementedError('grow-diag-final: Sorry for incovinience')
        elif method == 'grow-diag-final-and':
            raise NotImplementedError('grow-diag-final-and: Sorry for incovinience')
        else:
            message = """
invalid argument: method
valid assignment of method is `grow`, `grow-diag`, `grow-diag-final`, or `grow-diag-final-and`
            """
            print(message, file=sys.stderr)
            sys.exit(1)


def parse():
    parser = argparse.ArgumentParser(
        description='Extract Alignment from GIZA output',
        usage='\n %(prog)s source_file target_file s2t_alignment t2s_alignment'
              '\n %(prog)s -h'
    )
    parser.add_argument('source_file', help='source file path')
    parser.add_argument('target_file', help='target file path')
    parser.add_argument('s2t_alignment_file', help='source to target alignment file path')
    parser.add_argument('t2s_alignment_file', help='target to source alignment file path')
    parser.add_argument('--alignment_method', default="intersection", type=str, help="alignment method")
    parser.add_argument('--output', default="giza.f2e", type=str, help='bi-directional alignment file')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse()
    sentences = []
    # アラインメント結果の格納
    with codecs.open(args.source_file, 'r') as src_f, codecs.open(args.target_file, 'r') as tgt_f, \
            codecs.open(args.s2t_alignment_file, 'r') as s2t_align, \
            codecs.open(args.t2s_alignment_file, 'r') as t2s_align:
        for s_line, t_line, s2t_align_line, t2s_align_line in zip(src_f, tgt_f, s2t_align, t2s_align):
            sentences.append(AlignmentSentences(s_line.strip(),
                                                t_line.strip(),
                                                s2t_align_line.strip(),
                                                t2s_align_line.strip()))

    with codecs.open(args.output, 'w', 'utf-8') as f_out:
        for i, sentence in enumerate(sentences):
            print("# Sentence pair:", i+1, file=f_out)
            print(sentence.src_snt, file=f_out)
            # アライメント結果を取得
            align = [[] for _ in range(len(sentence.tgt_snt.split(' ')))]
            intersection = sentence.extract_align(method=args.alignment_method)
            for s_idx, t_idx in intersection:
                align[t_idx].append(s_idx)
            print('NULL ({ }) ', end='', file=f_out)  # NULLアライメント（mosesでのアラインメントとの互換性のため）
            print(' '.join([t_word+' ({ '+' '.join(str(a + 1) for a in t_a)+' })'
                            for t_word, t_a in zip(sentence.tgt_snt.split(' '), align)]), file=f_out)

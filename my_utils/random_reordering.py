#!/usr/bin/python3
# -*- coding: utf-8 -*-

import gzip
import re
import sys


def main():
    file_path = sys.argv[1]
    output_path = sys.argv[2]
    with gzip.open(file_path, 'rb', 'utf-8') as alignfile, open(output_path, 'w') as fout:
        for _ in alignfile:
            source = alignfile.readline().strip().decode('utf-8')
            source_aligns = [(s.strip(), None) for s in source.split()]
            target_aligns = alignfile.readline().strip().decode('utf-8')
            t_words_aligns = re.split('\(\{|\}\)', target_aligns)[:-1]
            target_aligns = [t_word_align.strip() for i, t_word_align in enumerate(t_words_aligns) if i % 2 == 1]
            for i, t_as in enumerate(target_aligns):
                for t_a in t_as.split(' '):
                    if t_a != '':
                        source_aligns[int(t_a) - 1] = (source_aligns[int(t_a) - 1][0], int(i))
            is_null_alignment = False
            tmp_sources = []
            for source_align in source_aligns:
                if source_align[1] is None:
                    if is_null_alignment:
                        tmp_sources[-1] = (tmp_sources[-1][0] + source_align[0], None)
                    else:
                        tmp_sources.append(source_align)
                        is_null_alignment = True
                else:
                    if is_null_alignment:
                        tmp_sources[-1] = (tmp_sources[-1][0] + source_align[0], source_align[1])
                        is_null_alignment = False
                    else:
                        tmp_sources.append(source_align)
            print(' '.join(s_a[0] for s_a in sorted(tmp_sources, key=lambda x: x[1])), file=fout)


if __name__ == '__main__':
    main()

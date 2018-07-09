#!/usr/bin/python3
# -*- coding: utf-8 -*-


def kendall_tau(align):
    """
    ケンダールのτの計算、順位相関係数となるようにしたもの
    :param align: アライメント先を表すリスト
    """
    if len(align) == 1 or len(align) == 0:
        return 0.0
    inc = 0
    for i in range(len(align) - 1):
        for j in range(i + 1, len(align)):
            if align[i] <= align[j]:
                inc += 1
    return 2 * inc / (len(align) * (len(align) - 1) / 2) - 1


def fuzzy_reordering_score(align, base=0):
    if len(align) == 0:
        return 0.0
    B = 0
    B += 1 if align[0] == base else 0
    B += 1 if align[-1] == max(align) else 0
    for i in range(len(align[:-1])):
        if align[i] == align[i + 1] or align[i] + 1 == align[i + 1]:
            B += 1
    return B / (len(align) + 1)


if __name__ == '__main__':
    print("***** rank correlation coefficient ***")
    print("  test_list = [5, 1, 2, 4, 3]  ")
    print("  kendall's tau         : %.4f" % kendall_tau([5, 1, 2, 4, 3]))
    print("  fuzzy reordering score: %.4f" % fuzzy_reordering_score([5, 1, 2, 4, 3]))
    print("**************************************")

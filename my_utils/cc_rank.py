#!/usr/bin/python3
# -*- coding: utf-8 -*-


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


def spearman_rho(align):
    raise NotImplementedError("Sorry for inconvenient...")


if __name__ == '__main__':
    print("***** rank correlation coefficient ***")
    print("  test_list = [5, 1, 2, 4, 3]  ")
    print("  kendall's tau : %.4f" % kendall_tau([5, 1, 2, 4, 3]))
    print("  spearman's rho: %.4f" % spearman_rho([5, 1, 2, 4, 3]))
    print("**************************************")

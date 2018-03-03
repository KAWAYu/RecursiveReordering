#!/usr/bin/python
# -*- coding: utf-8 -*-

from chainer import Chain
import chainer.links as L
import chainer.functions as F


# シンプルなモデル（単語エンベッディングのみ）
class RecursiveNet(Chain):
    def __init__(self, n_vocab, n_embed, n_units, n_labels):
        super(RecursiveNet, self).__init__(
            emb=L.EmbedID(n_vocab, n_embed),
            el=L.Linear(n_embed, n_units),
            l=L.Linear(n_units*2, n_units),
            w=L.Linear(n_units, n_labels),
        )

    def leaf(self, x):
        return F.relu(self.el(F.relu(self.emb(x))))

    def node(self, left, right):
        return F.relu(self.l(F.concat((left, right))))

    def label(self, v):
        return self.w(v)


class RecursivePosNet(Chain):
    def __init__(self, n_pos, n_pos_embed, n_units, n_labels):
        super(RecursivePosNet, self).__init__(
            pos_emb=L.EmbedID(n_pos, n_pos_embed),
            el=L.Linear(n_pos_embed, n_units),
            l=L.Linear(n_units*2 + n_pos_embed, n_units),
            w=L.Linear(n_units, n_labels),
        )

    def leaf(self, p):
        return F.relu(self.el(F.relu(self.pos_emb(p))))

    def node(self, left, right, p):
        return F.relu(self.l(F.concat((F.concat((left, right)), F.relu(self.pos_emb(p))))))

    def label(self, v):
        return self.w(v)


# 品詞エンベッディングを結合するモデル
class RecursiveNet_CatPos(Chain):
    def __init__(self, n_vocab, n_pos, n_embed, n_pos_embed, n_units, n_labels):
        super(RecursiveNet_CatPos, self).__init__(
            pos_emb=L.EmbedID(n_pos, n_pos_embed),
            emb=L.EmbedID(n_vocab, n_embed),
            pel=L.Linear(n_embed + n_pos_embed, n_units),
            l=L.Linear(n_units*2 + n_pos_embed, n_units),
            w=L.Linear(n_units, n_labels)
        )

    def leaf(self, x, p):
        return F.relu(self.pel(F.concat((F.relu(self.emb(x)), F.relu(self.pos_emb(p))))))

    def node(self, left, right, p):
        return F.relu(self.l(F.concat((F.concat((left, right)), F.relu(self.pos_emb(p))))))

    def label(self, v):
        return self.w(v)


# 品詞エンベッディングを足し合わせるモデル
class RecursiveNet_AddPos(Chain):
    def __init__(self, n_vocab, n_pos, n_embed, n_pos_embed, n_units, n_labels):
        super(RecursiveNet_AddPos, self).__init__(
            pos_emb=L.EmbedID(n_pos, n_pos_embed),
            emb=L.EmbedID(n_vocab, n_embed),
            el=L.Linear(n_embed, n_units),
            pl=L.Linear(n_pos_embed, n_units),
            l=L.Linear(n_units*2, n_units),
            w=L.Linear(n_units, n_labels),
        )

    def leaf(self, x, p):
        # 葉ノード
        pos_embed = F.relu(self.pl(F.relu(self.pos_emb(p))))
        word_emb = F.relu(self.el(F.relu(self.emb(x))))
        return F.relu(self.pel(word_emb + pos_embed))

    def node(self, left, right, p):
        pos_embed = F.relu(self.pos_emb(p))
        return F.relu(self.l(F.concat((left, right))) + pos_embed)

    def label(self, v):
        return self.w(v)


class RecursiveNet_Child(Chain):
    def __init__(self, n_vocab, n_pos, n_embed, n_pos_embed, n_units, n_labels):
        super(RecursiveNet_Child, self).__init__(
            pos_emb=L.EmbedID(n_pos, n_pos_embed),
            emb=L.mbedID(n_vocab, n_embed),
            pel=L.Linear(n_embed + n_pos_embed, n_units),
            l=L.Linear(n_units*2 + n_pos_embed, n_units),
            w=MLP(n_units, n_pos_embed, n_labels),
        )

    def leaf(self, x, p):
        return F.relu(self.pel(F.concat((F.relu(self.emb(x)), F.relu(self.pos_emb(p))))))

    def node(self, left, right, p):
        return F.relu(self.l(F.concat((F.concat((left, right)), F.relu(self.pos_emb(p))))))

    def label(self, left, right, p):
        return self.w(left, right, F.relu(self.pos_emb(p)))


class MLP(Chain):
    def __init__(self, n_units, n_pos_embed, n_labels):
        super(MLP, self).__init__(
            l1=L.Linear(n_units*2 + n_pos_embed, n_units),
            l2=L.Linear(n_units, n_labels),
        )

    def __call__(self, left, right, p):
        l1_out = F.relu(self.l1(F.concat((F.concat((left, right)), p))))
        return F.relu(self.l2(l1_out))

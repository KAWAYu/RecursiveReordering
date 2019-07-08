#!/usr/bin/python
# -*- coding: utf-8 -*-

from chainer import Chain
import chainer.links as L
import chainer.functions as F


# シンプルなモデル（単語エンベッディングのみ）
class RecursiveNet(Chain):
    def __init__(self, n_vocab, n_embed, n_units, n_labels):
        super(RecursiveNet, self).__init__()
        with self.init_scope():
            self.emb = L.EmbedID(n_vocab, n_embed)
            self.el = L.Linear(n_embed, n_units)
            self.l = L.Linear(n_units*2, n_units)
            self.w = L.Linear(n_units, n_labels)

    def leaf(self, x):
        return F.relu(self.el(F.relu(self.emb(x))))

    def node(self, left, right):
        return F.relu(self.l(F.concat((left, right))))

    def label(self, v):
        return self.w(v)


class RecursivePosNet(Chain):
    def __init__(self, n_pos, n_pos_embed, n_units, n_labels):
        super(RecursivePosNet, self).__init__()
        with self.init_scope():
            self.pos_emb = L.EmbedID(n_pos, n_pos_embed)
            self.el = L.Linear(n_pos_embed, n_units)
            self.l = L.Linear(n_units*2 + n_pos_embed, n_units)
            self.w = L.Linear(n_units, n_labels)

    def leaf(self, p):
        return F.relu(self.el(F.relu(self.pos_emb(p))))

    def node(self, left, right, p):
        return F.relu(self.l(F.concat((F.concat((left, right)), F.relu(self.pos_emb(p))))))

    def label(self, v):
        return self.w(v)


# 品詞エンベッディングを結合するモデル
class RecursiveNet_CatPos(Chain):
    def __init__(self, n_vocab, n_pos, n_embed, n_pos_embed, n_units, n_labels):
        super(RecursiveNet_CatPos, self).__init__()
        with self.init_scope():
            self.pos_emb = L.EmbedID(n_pos, n_pos_embed)
            self.emb = L.EmbedID(n_vocab, n_embed)
            self.pel = L.Linear(n_embed + n_pos_embed, n_units)
            self.l = L.Linear(n_units*2 + n_pos_embed, n_units)
            self.w = L.Linear(n_units, n_labels)

    def leaf(self, x, p):
        return F.relu(self.pel(F.concat((F.relu(self.emb(x)), F.relu(self.pos_emb(p))))))

    def node(self, left, right, p):
        return F.relu(self.l(F.concat((F.concat((left, right)), F.relu(self.pos_emb(p))))))

    def label(self, v):
        return self.w(v)


# 品詞エンベッディングを足し合わせるモデル
class RecursiveNet_AddPos(Chain):
    def __init__(self, n_vocab, n_pos, n_embed, n_pos_embed, n_units, n_labels):
        super(RecursiveNet_AddPos, self).__init__()
        with self.init_scope():
            self.pos_emb = L.EmbedID(n_pos, n_pos_embed)
            self.emb = L.EmbedID(n_vocab, n_embed)
            self.el = L.Linear(n_embed, n_units)
            self.pl = L.Linear(n_pos_embed, n_units)
            self.l = L.Linear(n_units*2, n_units)
            self.w = L.Linear(n_units, n_labels)

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
        super(RecursiveNet_Child, self).__init__()
        with self.init_scope():
            self.pos_emb = L.EmbedID(n_pos, n_pos_embed)
            self.emb = L.EmbedID(n_vocab, n_embed)
            self.pel = L.Linear(n_embed + n_pos_embed, n_units)
            self.l = L.Linear(n_units*2 + n_pos_embed, n_units)
            self.w = MLP(n_units, n_pos_embed, n_labels)

    def leaf(self, x, p):
        return F.relu(self.pel(F.concat((F.relu(self.emb(x)), F.relu(self.pos_emb(p))))))

    def node(self, left, right, p):
        return F.relu(self.l(F.concat((F.concat((left, right)), F.relu(self.pos_emb(p))))))

    def label(self, left, right, p):
        return self.w(left, right, F.relu(self.pos_emb(p)))


class MLP(Chain):
    def __init__(self, n_units, n_pos_embed, n_labels):
        super(MLP, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(n_units*2 + n_pos_embed, n_units)
            self.l2 = L.Linear(n_units, n_labels)

    def __call__(self, left, right, p):
        l1_out = F.relu(self.l1(F.concat((F.concat((left, right)), p))))
        return F.relu(self.l2(l1_out))


class RecursiveNetDev(Chain):
    def __init__(self, n_vocab, n_pos, n_embed, n_pos_embed, n_units, n_labels):
        super(RecursiveNetDev, self).__init__()
        with self.init_scope():
            self.emb = L.EmbedID(n_vocab, n_embed)
            self.pos_emb = L.EmbedID(n_pos, n_pos_embed)
            self.pel = L.Linear(n_embed + n_pos_embed, n_units)
            self.l = L.Linear(n_units * 2 + n_pos_embed, n_units)
            self.w1 = L.Linear(n_units * 6 + n_pos_embed, n_units * 3)
            self.w = L.Linear(n_units * 3, n_labels)

    def leaf(self, x, p):
        return F.relu(self.pel(F.concat((F.relu(self.emb(x)), F.relu(self.pos_emb(p))))))

    def node(self, left, right, p):
        return F.relu(self.l(F.concat((F.concat((left, right)), F.relu(self.pos_emb(p))))))

    def label(self, left, right, left_l, left_r, right_l, right_r, p):
        left_concat = F.concat((F.relu(self.emb(left_l)), F.relu(self.emb(left_r))))
        right_concat = F.concat((F.relu(self.emb(right_l)), F.relu(self.emb(right_r))))
        node_concat = F.concat((F.relu(self.emb(left)), F.relu(self.emb(right))))
        node_v = F.concat((F.concat((left_concat, right_concat)), F.concat((node_concat, F.relu(self.pos_emb(p))))))
        return self.w(F.relu(self.w1(node_v)))

#!/usr/bin/python3
# encoding: utf-8

import argparse
import codecs
import collections
import copy
import json
import pickle
import random
from threading import Thread
import time

import matplotlib
import numpy as np
import chainer
from chainer import cuda, optimizers, Variable, serializers
import chainer.functions as F
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import Recursive_util_pos as util
from Recursive_model import RecursiveNet_CatPos as RecursiveNet

# xp = numpy or cuda.cupy
xp = np


def parse():
    parser = argparse.ArgumentParser(
        description='Reordering with Recursive Neural Network',
        usage='\n %(prog)s train_file align dev_file align reorder_files [options]'
              '\n %(prog)s -h'
    )

    parser.add_argument('filepath', help='training file path')
    parser.add_argument('alignmentfile', help='alignment file(format ".gz")')
    parser.add_argument('devlop_file', default="", type=str,
                        help='development file path')
    parser.add_argument('devlop_alignment', default="", type=str,
                        help='development alignment file path')
    parser.add_argument('reorderfile', nargs='*',
                        help='file path you want to reorder')
    parser.add_argument('--vocab_pkl', default="", type=str,
                        help='vocabulary pickle file')
    parser.add_argument('--vocab_size', default=-1, type=int,
                        help='the max number of vocabulary')
    parser.add_argument('--epoch', '-e', default=100, type=int,
                        help='number of epoch to train')
    parser.add_argument('--unit', '-u', default=30, type=int,
                        help='number of units')
    parser.add_argument('--embed_size', '-emb', default=30, type=int,
                        help='number of embedding size')
    parser.add_argument('--pos_embed_size', '-pemb', default=30, type=int,
                        help='number of pos-tag embedding size')
    parser.add_argument('--batchsize', '-b', default=128, type=int,
                        help='size of minibatch')
    parser.add_argument('--label', '-l', default=2, type=int,
                        help='number of labels')
    parser.add_argument('--evalinterval', '-p', default=5, type=int,
                        help='number of epochs per evaluation')
    parser.add_argument('--gpus', '-g', default=-1, type=int,
                        help='number of gpu you want to use')
    parser.add_argument('--model', '-m', default=-1, type=int,
                        help='appoint the epoch if you start from previous training')
    parser.add_argument('--img_name', default="", type=str,
                        help='image\'s name of learning curve')
    parser.add_argument('--optimize', default='adam', type=str,
                        help='optimizer for neural network(default: Adam)')
    parser.add_argument('--tree_type', default="enju", type=str,
                        help='structure of tree')
    parser.add_argument('--visualize', '-v', dest='visualize', action='store_true')
    parser.set_defaults(visualize=False)
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


def traverse_dev(m, dev_trees, loss_list, gpu_num):
    if gpu_num >= 0:
        cuda.get_device_from_id(gpu_num).use()
        m.to_gpu()
    dev_batch_loss = 0
    for tree in dev_trees:
        loss, _ = traverse(m, tree, train=True, pred=False)
        dev_batch_loss += loss.data.tolist()
    loss_list.append(dev_batch_loss / len(dev_trees))


def evaluate(model, eval_trees):
    m = model.copy()
    result = collections.defaultdict(lambda: 0)
    pred_lists = []
    sum_loss = 0
    for tree in eval_trees:
        loss, pred_list = traverse(m, tree, train=True, pred=False, evaluate=result)
        pred_lists.append(pred_list)
        sum_loss += loss.data
    acc_node = 100.0*result['correct_node'] / result['total_node']
    print('  Node accuracy: {0:.2f} %% ({1:,d}/{2:,d})'.format(
      acc_node, result['correct_node'], result['total_node']))
    return acc_node, pred_lists, sum_loss


if __name__ == '__main__':
    args = parse()

    loss_curve = []
    acc_curve = []
    dev_loss = []
    dev_batch_loss = []
    train_batch_loss = []

    vocab = {'<UNK>': 0}
    cat_vocab = {'<UNK>': 0}
    max_size = None

    print('Reading training data...')
    train_trees, vocab = util.read_train(args.filepath, args.alignmentfile, vocab, max_size, args.vocab_size, cat_vocab, args.tree_type)
    print('Reading development data...')
    dev_trees = util.read_dev(args.devlop_file, args.devlop_alignment, vocab, cat_vocab, args.tree_type)

    if args.vocab_pkl:
        pickle.dump(vocab, open(args.vocab_pkl, 'wb'))
        pickle.dump(cat_vocab, open(args.vocab_pkl + '.pos', 'wb'))

    rtrees = []
    for fp in args.reorderfile:
        rtrees.append(util.read_reorder(fp, vocab, cat_vocab, args.tree_type))

    print('Creating model...')
    model = RecursiveNet(len(vocab), len(cat_vocab), args.embed_size, args.pos_embed_size, args.unit, args.label)
    start_epoch = 0
    if args.model != -1:  # 保存したモデルがある場合
        serializers.load_hdf5('epoch_' + str(args.model) + '.model', model)
        start_epoch = args.model

    if args.gpus >= 0:  # GPU使用の設定
        cuda.get_device_from_id(args.gpus).use()
        model.to_gpu(args.gpus)
        xp = cuda.cupy

    if args.optimize == 'sgd':  # 学習率のアルゴリズム
        optm = optimizers.SGD()
    elif args.optimize == 'adagrad':
        optm = optimizers.AdaGrad()
    elif args.optimize == 'adam':
        optm = optimizers.Adam()
    elif args.optimize == 'adadelta':
        optm = optimizers.AdaDelta()
    elif args.optimize == 'rmsprop':
        optm = optimizers.RMSprop()
    optm.setup(model)
    optm.add_hook(chainer.optimizer.WeightDecay(0.0001))
    optm.add_hook(chainer.optimizer.GradientClipping(5))

    start_time = time.time()
    cur_time = start_time

    print('Training start...')
    for epoch in range(start_epoch, args.epoch):
        train_batch_loss = []
        dev_batch_loss = []
        batch_loss = sum_loss = 0
        print('Epoch: {0:d}'.format(epoch+1))
        random.shuffle(train_trees)
        cur_time = time.time()
        for t_num, tree in enumerate(train_trees):
            loss, _ = traverse(model, tree, train=True)
            batch_loss += loss
            if (t_num+1) % args.batchsize == 0:
                print('%d trees are learned' % (t_num+1))
                batch_loss /= args.batchsize
                train_batch_loss.append(batch_loss.data.tolist())
                model.cleargrads()
                batch_loss.backward()
                optm.update()
                batch_loss = 0
                print("Development data evaluation by batch:")
                # 検証データの評価
                Thread(target=traverse_dev, args=(copy.deepcopy(model), dev_trees, dev_batch_loss, args.gpus)).start()
            sum_loss += float(loss.data)  # epochでの総和
        if (t_num + 1) % args.batchsize != 0:
            batch_loss /= (t_num + 1) % args.batchsize
            train_batch_loss.append(batch_loss.data.tolist())
            model.cleargrads()
            batch_loss.backward()
            optm.update()
            # 検証データの評価
            Thread(target=traverse_dev, args=(copy.deepcopy(model), dev_trees, dev_batch_loss, args.gpus)).start()

        now = time.time()
        json.dump({"train_batch": train_batch_loss, 'dev_batch': dev_batch_loss}, 
                  open('epoch%d_loss_by_tree.json' % (epoch + 1), 'w'))
        loss_curve.append(sum_loss / len(train_trees))
        print('train loss: {:.2f}'.format(sum_loss / len(train_trees)))

        print("Development data evaluation:")
        Thread(target=traverse_dev, args=(copy.deepcopy(model), dev_trees, dev_loss, args.gpu)).start()
        # d_loss = 0
        # for dev_tree in dev_trees:
        #    loss, _ = traverse(model, dev_tree, train=True, pred=False)
        #    d_loss += loss.data.tolist()
        # dev_loss.append(d_loss / len(dev_trees))
        # print('dev loss: {:.2f}'.format(d_loss / len(dev_trees)))

        throughput = float(len(train_trees)) / (now - cur_time)
        print('{:.2f} iter/sec, {:.2f} sec'.format(throughput, now-cur_time))
        print()

        if (epoch+1) % args.evalinterval == 0:
            print("Model saving...")
            serializers.save_hdf5('./epoch_'+str(epoch+1)+'.model', model)
        
        # バッチのロスの描画
        plt.clf()
        plt.plot(np.array([(i+1)*args.batchsize for i in range(len(train_batch_loss))]), np.array(train_batch_loss),
                 label="train batch")
        plt.plot(np.array([(i+1)*args.batchsize for i in range(len(dev_batch_loss))]), np.array(dev_batch_loss),
                 label="dev batch")
        plt.title("%d's epoch batch loss curve" % (epoch + 1))
        plt.ylim(0, 15)
        plt.ylabel("loss")
        plt.xlabel("batch(*%d trees)" % args.batchsize)
        plt.legend()
        plt.savefig("%ds_epoch_loss_curve.png" % (epoch + 1))
    
    json.dump({"loss": dev_loss}, open('dev_loss_by_epoch.json', 'w'))
     
    for i, fp in enumerate(args.reorderfile):
        with codecs.open(fp+'.reordered', 'w', 'utf-8') as fre:
            for tree in rtrees[i]:
                _, pred = traverse(model, tree, train=False, pred=True)
                print(' '.join(pred), file=fre)

    # エポックごとのロスの描画
    plt.clf()
    plt.figure(figsize=(8, 8))
    plt.plot(np.array([i+1 for i in range(args.epoch)]), np.array(loss_curve), label="train")
    plt.plot(np.array([i+1 for i in range(args.epoch)]), np.array(dev_loss), label="dev")
    higher = max(loss_curve + dev_loss)
    lower = min(loss_curve + dev_loss)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.ylim(lower-0.3, higher+0.3)
    plt.legend()
    if args.visualize:
        plt.show()
    else:
        if args.img_name:
            plt.savefig(args.img_name)

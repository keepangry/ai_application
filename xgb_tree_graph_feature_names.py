#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/4/3 3:16 PM
# @Author  : yangsen
# @Site    : 
# @File    : to_graphviz.py
# @Software: PyCharm

import matplotlib.pyplot as plt
import re
from graphviz import Digraph
import numpy as np
import xgboost as xgb
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

_NODEPAT = re.compile(r'(\d+):\[(.+)\]')
_LEAFPAT = re.compile(r'(\d+):(leaf=.+)')
_EDGEPAT = re.compile(r'yes=(\d+),no=(\d+),missing=(\d+)')
_EDGEPAT2 = re.compile(r'yes=(\d+),no=(\d+)')


def map_node_name(label, feat_map):
    feature, value = label.split('<')
    return "%s<%s" % (feat_map[feature], str(round(float(value),3)))


def get_feature_map(feature_names):
    result = dict(zip(range(len(feature_names)), feature_names))
    return {'f%s' % k: v for k, v in result.items()}


def _parse_node(graph, text, feat_map):
    """parse dumped node"""
    match = _NODEPAT.match(text)
    if match is not None:
        node = match.group(1)
        label = map_node_name(match.group(2), feat_map)
        graph.node(node, label=label, shape='circle')
        return node
    match = _LEAFPAT.match(text)
    if match is not None:
        node = match.group(1)
        graph.node(node, label=match.group(2), shape='box')
        return node
    raise ValueError('Unable to parse node: {0}'.format(text))


def _parse_edge(graph, node, text, yes_color='#0000FF', no_color='#FF0000'):
    """parse dumped edge"""
    try:
        match = _EDGEPAT.match(text)
        if match is not None:
            yes, no, missing = match.groups()
            if yes == missing:
                graph.edge(node, yes, label='yes, missing', color=yes_color)
                graph.edge(node, no, label='no', color=no_color)
            else:
                graph.edge(node, yes, label='yes', color=yes_color)
                graph.edge(node, no, label='no, missing', color=no_color)
            return
    except ValueError:
        pass
    match = _EDGEPAT2.match(text)
    if match is not None:
        yes, no = match.groups()
        graph.edge(node, yes, label='yes', color=yes_color)
        graph.edge(node, no, label='no', color=no_color)
        return
    raise ValueError('Unable to parse edge: {0}'.format(text))


def to_graphviz(booster, feature_names, num_trees=0):
    feat_map = get_feature_map(feature_names)

    tree = booster.get_dump()[num_trees]
    tree = tree.split()

    yes_color = '#0000FF'
    no_color = '#FF0000'
    graph = Digraph()
    for i, text in enumerate(tree):
        if text[0].isdigit():
            node = _parse_node(graph, text, feat_map)
        else:
            if i == 0:
                # 1st string must be node
                raise ValueError('Unable to parse given string as tree')
            _parse_edge(graph, node, text, yes_color=yes_color,
                        no_color=no_color)
    return graph


def feature_importance(booster, feature_names, max_num_features=10, importance_type='weight'):
    feat_map = get_feature_map(feature_names)
    importance = booster.get_score(importance_type=importance_type)
    tuples = [(feat_map[k], importance[k]) for k in importance]
    tuples = sorted(tuples, key=lambda x: x[1], reverse=True)[:max_num_features]
    return tuples


if __name__ == "__main__":
    iris = load_iris()
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)
    params = {'objective': 'multi:softprob', 'num_class': 3}
    # feature_names=iris.feature_names 如果不填写会是f0,f1
    dtrain = xgb.DMatrix(X_train, y_train, feature_names=iris.feature_names)

    bst = xgb.train(params, dtrain, num_boost_round=20)
    booster = bst

    # 直接使用，特征显示为f0,f1,f2。
    """
        特别是直接从线上spark训练的模型
    """
    xgb.plot_importance(bst, max_num_features=20)
    # plt.show()
    feature_names = iris.feature_names
    # xgb.to_graphviz()

    importance = feature_importance(booster, feature_names, max_num_features=20, importance_type='gain')
    list(map(print, importance))

    graph = to_graphviz(bst, feature_names, num_trees=0)
    graph.render(directory=r'./graphs/', filename='result-20190228-1')

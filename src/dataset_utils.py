#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8

# import torch
# import ipdb
# import math
import ipdb
import dgl

# import os
import os.path as osp
import numpy as np
import torch.nn.functional as F
import torch_geometric.transforms as T

# from cSBM_dataset import dataset_ContextualSBM
# from torch_geometric.datasets import Planetoid
# from torch_geometric.datasets import Coauthor
# from torch_geometric.datasets import Amazon
from torch_geometric.nn import APPNP
from torch_sparse import coalesce
from torch_geometric.data import InMemoryDataset, download_url, Data
from torch_geometric.utils.undirected import is_undirected, to_undirected
from torch_geometric.io import read_npz


from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset, AmazonCoBuyComputerDataset, AmazonCoBuyPhotoDataset
from torch_geometric.datasets import WikipediaNetwork, WebKB, Actor

def preprocess_data(name):
    # assert name in ['cSBM_data_Aug_19_2020-13:06',
    #                 'cSBM_data_Aug_18_2020-18:50',
    #                 'cSBM_data_Aug_21_2020-10:06',
    #                 'cSBM_data_Aug_19_2020-20:41',
    #                 'cSBM_data_Aug_21_2020-11:04',
    #                 'cSBM_data_Aug_21_2020-11:21',
    #                 'cSBM_data_Sep_01_2020-14:15',
    #                 'cSBM_data_Sep_01_2020-14:18',
    #                 'cSBM_data_Sep_01_2020-14:19',
    #                 'cSBM_data_Sep_01_2020-14:32',
    #                 'cSBM_data_Sep_01_2020-14:22',
    #                 'cSBM_data_Sep_01_2020-14:23',
    #                 'cSBM_data_Sep_01_2020-14:27',
    #                 'cSBM_data_Sep_01_2020-14:29',
    #                 'Cora', 'Citeseer', 'PubMed',
    #                 'Computers', 'Photo',
    #                 'chameleon', 'film', 'squirrel',
    #                 'Texas', 'Cornell']

    # if name in ['cSBM_data_Aug_19_2020-13:06',
    #             'cSBM_data_Aug_18_2020-18:50',
    #             'cSBM_data_Aug_21_2020-10:06',
    #             'cSBM_data_Aug_19_2020-20:41',
    #             'cSBM_data_Aug_21_2020-11:04',
    #             'cSBM_data_Aug_21_2020-11:21',
    #             'cSBM_data_Sep_01_2020-14:15',
    #             'cSBM_data_Sep_01_2020-14:18',
    #             'cSBM_data_Sep_01_2020-14:19',
    #             'cSBM_data_Sep_01_2020-14:32',
    #             'cSBM_data_Sep_01_2020-14:22',
    #             'cSBM_data_Sep_01_2020-14:23',
    #             'cSBM_data_Sep_01_2020-14:27',
    #             'cSBM_data_Sep_01_2020-14:29']:
    # if 'cSBM_data' in name:
    #     path = '../data/'
    #     dataset = dataset_ContextualSBM(path, name=name)
    # else:
    #     name = name.lower()

    if name in ['cora', 'citeseer', 'pubmed']:
        if name == 'cora':
            dataset = CoraGraphDataset(verbose=False)
        elif name == 'citeseer':
            dataset = CiteseerGraphDataset(verbose=False)
        elif name == 'pubmed':
            dataset = PubmedGraphDataset(verbose=False)
        graph = dataset[0]

    elif name in ['computers', 'photo']:
        if name == 'computers':
            dataset = AmazonCoBuyComputerDataset(verbose=False)
        elif name == 'photo':
            dataset = AmazonCoBuyPhotoDataset(verbose=False)
        graph = dataset[0]

    elif name in ['chameleon', 'squirrel']:
        dataset = WikipediaNetwork(
            root='../data/', name=name, transform=T.NormalizeFeatures())
        g = dataset[0]
        src = g.edge_index[0]
        dst = g.edge_index[1]
        graph = dgl.graph((src,dst), num_nodes=g.x.size(0))
        graph.ndata['feat'] = g.x
        graph.ndata['label'] = g.y
    
    elif name in ['film']:
        dataset = Actor(
            root='../data/', transform=T.NormalizeFeatures())
        g = dataset[0]
        src = g.edge_index[0]
        dst = g.edge_index[1]
        graph = dgl.graph((src,dst), num_nodes=g.x.size(0))
        graph.ndata['feat'] = g.x
        graph.ndata['label'] = g.y
    
    elif name in ['texas', 'cornell']:
        dataset = WebKB(root='../data/',
                        name=name, transform=T.NormalizeFeatures())
        g = dataset[0]
        src = g.edge_index[0]
        dst = g.edge_index[1]
        graph = dgl.graph((src,dst), num_nodes=g.x.size(0))
        graph.ndata['feat'] = g.x
        graph.ndata['label'] = g.y

    else:
        raise ValueError(f'dataset {name} not supported in dataloader')

    return graph

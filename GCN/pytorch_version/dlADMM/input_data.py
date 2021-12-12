import torch.nn.functional as F
from torch_geometric.datasets import Planetoid, Amazon, Coauthor,Flickr
import math
from torch_geometric.nn.inits import glorot

import torch
from torch.nn import Parameter
from torch_scatter import scatter_add
import torch_sparse
from torch_sparse import SparseTensor, fill_diag, sum, mul, cat
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes
from dlADMM.common import gcn_norm
import numpy as np
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
import copy
import torch_geometric.transforms as T

# =================== onehot label ===================
def onehot(label, num_classes=None):
    """
    return the onehot label for mse loss
    """
    if num_classes == None:
        classes = set(np.unique(label.detach().numpy()))
    else:
        classes = set(np.linspace(0, num_classes-1, num_classes))
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    x = list(map(classes_dict.get, label.detach().numpy()))
    label_onehot = np.array(x)
    label_onehot = torch.tensor(label_onehot, dtype=torch.float)
    return label_onehot


class cora():
    def __init__(self):
        try:
            self.data = Planetoid(root='/home/xd/Documents/code/admm_gnn/dataset/cora', name='cora')[0]
            self.processed_dir = Planetoid(root='/home/xd/Documents/code/admm_gnn/dataset/cora', name='cora').processed_dir

        except:
            self.data = Planetoid(root='/tmp/cora', name='cora')[0]
            self.processed_dir = Planetoid(root='/tmp/cora', name='cora').processed_dir
        self.x =self.data.x
        self.label = self.data.y
        self.num_classes = max(self.label) + 1
        self.num_features = self.data.x.size()[1]
        self.label_train, self.label_test = self.data.y[self.data.train_mask], self.data.y[self.data.test_mask]
        self.label_train_onehot = onehot(self.label_train, num_classes=self.num_classes)
        self.adj = gcn_norm(self.data.edge_index)
        # self.x_train, self.x_test = self.data.x[self.data.train_mask], self.data.x[self.data.test_mask]
        self.train_mask, self.test_mask = self.data.train_mask, self.data.test_mask


class pubmed():
    def __init__(self):
        try:
            self.data = Planetoid(root='/home/xd/Documents/code/admm_gnn/dataset/PubMed', name='PubMed')[0]
            self.processed_dir = Planetoid(root='/home/xd/Documents/code/admm_gnn/dataset/PubMed', name='PubMed').processed_dir

        except:
            self.data = Planetoid(root='/tmp/PubMed', name='PubMed')[0]
            self.processed_dir = Planetoid(root='/tmp/PubMed', name='PubMed').processed_dir
        self.x =self.data.x
        self.label = self.data.y
        self.num_classes = max(self.label) + 1
        self.num_features = self.data.x.size()[1]
        self.label_train, self.label_test = self.data.y[self.data.train_mask], self.data.y[self.data.test_mask]
        self.label_train_onehot = onehot(self.label_train, num_classes=self.num_classes)
        self.adj = gcn_norm(self.data.edge_index)
        # self.x_train, self.x_test = self.data.x[self.data.train_mask], self.data.x[self.data.test_mask]
        self.train_mask, self.test_mask = self.data.train_mask, self.data.test_mask


class citeseer():
    def __init__(self):
        try:
            self.data = Planetoid(root='/home/xd/Documents/code/admm_gnn/dataset/citeseer', name='citeseer')[0]
            self.processed_dir = Planetoid(root='/home/xd/Documents/code/admm_gnn/dataset/citeseer', name='citeseer').processed_dir

        except:
            self.data = Planetoid(root='/tmp/citeseer', name='citeseer')[0]
            self.processed_dir = Planetoid(root='/tmp/citeseer', name='citeseer').processed_dir
        self.x =self.data.x
        self.label = self.data.y
        self.num_classes = max(self.label) + 1
        self.num_features = self.data.x.size()[1]
        self.label_train, self.label_test = self.data.y[self.data.train_mask], self.data.y[self.data.test_mask]
        self.label_train_onehot = onehot(self.label_train, num_classes=self.num_classes)
        self.adj = gcn_norm(self.data.edge_index)
        # self.x_train, self.x_test = self.data.x[self.data.train_mask], self.data.x[self.data.test_mask]
        self.train_mask, self.test_mask = self.data.train_mask, self.data.test_mask

class coauthor_cs():
    def __init__(self):
        try:
            self.data = Coauthor(root='/home/xd/Documents/code/admm_gnn/dataset/cs', name='cs')[0]
            self.processed_dir = Coauthor(root='/home/xd/Documents/code/admm_gnn/dataset/cs', name='cs').processed_dir

        except:
            self.data = Coauthor(root='/tmp/cs', name='cs')[0]
            self.processed_dir = Coauthor(root='/tmp/cs', name='cs').processed_dir
        self.x =self.data.x
        self.label = self.data.y
        self.num_classes = max(self.label) + 1
        self.num_features = self.data.x.size()[1]
        # split the dataset
        split_dataset(self)
        self.label_train, self.label_test = self.data.y[self.train_mask], self.data.y[self.test_mask]
        self.label_train_onehot = onehot(self.label_train, num_classes=self.num_classes)
        self.adj = gcn_norm(self.data.edge_index)
        # self.x_train, self.x_test = self.x[self.train_mask], self.x[self.test_mask]
class coauthor_physics():
    def __init__(self):
        try:
            self.data = Coauthor(root='/home/xd/Documents/code/admm_gnn/dataset/physics', name='physics')[0]
            self.processed_dir = Coauthor(root='/home/xd/Documents/code/admm_gnn/dataset/physics', name='cs').processed_dir

        except:
            self.data = Coauthor(root='/tmp/physics', name='physics')[0]
            self.processed_dir = Coauthor(root='/tmp/physics', name='physics').processed_dir
        self.x =self.data.x
        self.label = self.data.y
        self.num_classes = max(self.label) + 1
        self.num_features = self.data.x.size()[1]
        # split the dataset
        split_dataset(self)
        self.label_train, self.label_test = self.data.y[self.train_mask], self.data.y[self.test_mask]
        self.label_train_onehot = onehot(self.label_train, num_classes=self.num_classes)
        self.adj = gcn_norm(self.data.edge_index)
        # self.x_train, self.x_test = self.x[self.train_mask], self.x[self.test_mask]
        self.edge_index = self.data.edge_index
        self.num_edges = self.data.num_edges
def split_dataset(data):
    num_train_per_class = 100
    num_test = 1000
    data.train_mask = torch.zeros(size=(1, data.label.size()[0])).squeeze(dim=0).bool().fill_(False)
    data.test_mask = torch.zeros(size=(1, data.label.size()[0])).squeeze(dim=0).bool().fill_(False)

    for c in range(data.num_classes):
        idx = (data.label == c).nonzero().view(-1)
        torch.manual_seed(seed=100)
        idx = idx[torch.randperm(idx.size(0))[:num_train_per_class]]
        data.train_mask[idx] = True

    remaining = (~data.train_mask).nonzero().view(-1)
    torch.manual_seed(seed=100)
    remaining = remaining[torch.randperm(remaining.size(0))]
    data.test_mask[remaining[:num_test]] = True

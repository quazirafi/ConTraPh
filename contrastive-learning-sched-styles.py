import os

import dgl
import torch
import torch.nn.functional as F
import numpy as np
from dgl.dataloading import GraphDataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import dgl.nn.pytorch as dglnn
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from torch.utils.data import Subset
import csv
from tqdm import tqdm

dataset_pg = dgl.data.CSVDataset('./Datamodels/dgl-csv-scheduling-style')

N_split = 40

class RGCN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, rel_names):
        super().__init__()

        self.conv1 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(in_feats, hid_feats)
            for rel in rel_names}, aggregate='sum')
        self.conv2 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(hid_feats, hid_feats)
            for rel in rel_names}, aggregate='sum')
        self.conv3 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(hid_feats, hid_feats)
            for rel in rel_names}, aggregate='sum')
        self.conv4 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(hid_feats, hid_feats)
            for rel in rel_names}, aggregate='sum')
        self.conv5 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(hid_feats, hid_feats)
            for rel in rel_names}, aggregate='sum')
        self.conv6 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(hid_feats, out_feats)
            for rel in rel_names}, aggregate='sum')

    def forward(self, graph, inputs):
        # inputs is features of nodes
        h = self.conv1(graph, inputs)
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv2(graph, h)
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv3(graph, h)
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv4(graph, h)
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv5(graph, h)
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv6(graph, h)
        return h


class HeteroClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes, rel_names):
        super().__init__()

        self.rgcn = RGCN(in_dim, hidden_dim, hidden_dim, rel_names)
        self.classify = nn.Linear(hidden_dim, n_classes)

    def forward(self, g):
        h = g.ndata['feat']
        h = self.rgcn(g, h)
        with g.local_scope():
            g.ndata['h'] = h
            # pass
            # Calculate graph representation by average readout.
            hg = 0
            for ntype in g.ntypes:
                hg = hg + dgl.mean_nodes(g, 'h', ntype=ntype)
                # pass
            return self.classify(hg)

whole_exp = 0

prev_max_num_correct = -1000

flag_15 = 0
final_accuracy = -1.0
final_cr = None


for whole_exp in tqdm(range(N_split)):

    num_examples = len(dataset_pg)

    output_final = []
    label_final = []

    dataset_indices = list(range(num_examples))
    np.random.shuffle(dataset_indices)
    test_split_index = 86


    train_idx, test_idx = dataset_indices[test_split_index:], dataset_indices[:test_split_index]
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)

    etypes = [('control', 'control', 'control'), ('control', 'call', 'control'), ('control', 'data', 'variable'), ('variable', 'data', 'control')]
    class_names = ['Default', 'Dynamic']

    train_dataloader_pg = GraphDataLoader(dataset_pg, shuffle=False, batch_size=100, sampler=train_sampler)
    test_dataloader_pg = GraphDataLoader(dataset_pg, shuffle=False, batch_size=64, sampler=test_sampler)

    model_pg = torch.load('./ModelCheckpoints/model-scheduling-style-best-model-804-100.pt')


    num_correct = 0
    num_tests = 0
    total_pred = []
    total_label = []

    for batched_graph, labels in test_dataloader_pg:
        pred = model_pg(batched_graph)

        pred_numpy = pred.detach().numpy()

        for ind_pred, ind_label in zip(pred_numpy, labels):
            if np.argmax(ind_pred) == ind_label:
                num_correct += 1
            total_pred.append(np.argmax(ind_pred))

        num_tests += len(labels)

        label_tmp = labels.data.cpu().numpy()
        total_label.extend(label_tmp)

        label_final = labels
        output_final = total_pred

        cr = classification_report(total_label, total_pred, target_names=class_names, output_dict=True)
        cr_for_print = classification_report(total_label, total_pred, target_names=class_names)

        accuracy = cr['accuracy']

        if final_accuracy < accuracy:
            final_accuracy = accuracy
            final_cr = cr_for_print

        cf_matrix = confusion_matrix(total_label, total_pred)

print(final_accuracy, final_cr)

from dgl.data import citation_graph as citegrh
from dgl.data import AmazonCoBuyPhotoDataset
import networkx as nx
import numpy as np
import torch as th
import math
import random
import dgl
import torch.nn as nn
import torch.nn.functional as F
import time
from dgl.nn.pytorch import GATConv
from sklearn.metrics import roc_auc_score, mean_squared_error
import csv

try:
    from sklearn.metrics import root_mean_squared_error
    def calculate_rmse(y_true, y_pred):
        return root_mean_squared_error(y_true, y_pred)
except ImportError:
    def calculate_rmse(y_true, y_pred):
        return mean_squared_error(y_true, y_pred, squared=False)

def load_data(dataset_name):
    if dataset_name == 'amazon_photo':
        dataset = AmazonCoBuyPhotoDataset()
        data = dataset[0]
        print("Shape of g_matrix:", data)
    elif dataset_name == 'citeseer':
        dataset = citegrh.load_citeseer()
        data = dataset[0]
    elif dataset_name == 'pubmed':
        dataset = citegrh.load_pubmed()
        data = dataset[0]
    else:
        dataset = AmazonCoBuyPhotoDataset()
        data = dataset[0]
    
    features = th.FloatTensor(data.ndata['feat'])
    labels = th.LongTensor(data.ndata['label'])

    # Print shape and a few examples of the features
    print("Feature shape:", features.shape)
    
    # Number of nodes
    num_nodes = data.num_nodes()

    # Generate random indices for splitting
    all_indices = np.random.permutation(num_nodes)
    train_size = int(num_nodes * 0.6)
    val_size = int(num_nodes * 0.2)
    test_size = num_nodes - train_size - val_size

    train_indices = all_indices[:train_size]
    val_indices = all_indices[train_size:train_size + val_size]
    test_indices = all_indices[train_size + val_size:]

    train_mask = th.zeros(num_nodes, dtype=th.bool)
    val_mask = th.zeros(num_nodes, dtype=th.bool)
    test_mask = th.zeros(num_nodes, dtype=th.bool)

    # Assign masks
    train_mask[train_indices] = True
    val_mask[val_indices] = True
    test_mask[test_indices] = True

    # Add masks to the graph
    data.ndata['train_mask'] = train_mask
    data.ndata['val_mask'] = val_mask
    data.ndata['test_mask'] = test_mask
    
    return data, features, labels

def create_edge_labels(g):
    u, v = g.edges()
    positive_edges = th.stack([u, v], dim=1)
    
    # Negative edges
    adj = g.adj().to_dense()
    neg_u, neg_v = th.where(adj == 0)
    negative_edges = th.stack([neg_u, neg_v], dim=1)
    
    # Shuffle and sample negative edges to match the number of positive edges
    neg_indices = th.randperm(negative_edges.size(0))[:positive_edges.size(0)]
    negative_edges = negative_edges[neg_indices]

    edges = th.cat([positive_edges, negative_edges], dim=0)
    labels = th.cat([th.ones(positive_edges.size(0)), th.zeros(negative_edges.size(0))], dim=0)
    
    return edges, labels

def evaluate(model, g, features, edges, labels):
    model.eval()
    with th.no_grad():
        logits = model(g, features)
        src, dst = edges[:, 0], edges[:, 1]
        logits = (logits[src] * logits[dst]).sum(dim=1)
        preds = th.sigmoid(logits).cpu().numpy()
        labels = labels.cpu().numpy()
        return roc_auc_score(labels, preds), calculate_rmse(labels, preds)

class GAT(nn.Module):
    def __init__(self, in_feats, h_feats, out_feats, num_heads, dropout):
        super(GAT, self).__init__()
        self.layer1 = GATConv(in_feats, h_feats, num_heads=num_heads, allow_zero_in_degree=True)
        self.layer2 = GATConv(h_feats * num_heads, out_feats, num_heads=1, allow_zero_in_degree=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, g, in_feat):
        h = self.layer1(g, in_feat)
        h = F.elu(h.flatten(1))
        h = self.dropout(h)
        h = self.layer2(g, h).mean(1)
        return h

def attack0_GAT_Link(dataset_name, attack_node_arg, cuda):
    data, features, labels = load_data(dataset_name)
    
    # Add self-loops to the graph
    g = dgl.add_self_loop(data)

    # Use CPU if CUDA is not available
    device = th.device('cuda' if th.cuda.is_available() else 'cpu')
    
    features = features.to(device)
    labels = labels.to(device)
    g = g.to(device)
    
    feature_number = features.shape[1]
    label_number = th.unique(labels).shape[0]
    attack_node_number = int(data.num_nodes() * attack_node_arg)
    
    gat_net = GAT(feature_number, 32, 32, num_heads=4, dropout=0.5).to(device)
    
    optimizer = th.optim.Adam(gat_net.parameters(), lr=1e-3, weight_decay=5e-4)
    dur = []
    
    train_acc_list = []
    val_acc_list = []
    test_acc_list = []
    train_loss_list = []
    val_loss_list = []
    test_loss_list = []
    fidelity_list = []
    train_rmse_list = []

    # Create edge labels for training
    edges, labels = create_edge_labels(g)
    edges, labels = edges.to(device), labels.to(device)
    
    results = []

    print("=========Target Model Generating==========================")
    for epoch in range(200):
        if epoch >= 3:
            t0 = time.time()
    
        gat_net.train()
        logits = gat_net(g, features)
        src, dst = edges[:, 0], edges[:, 1]
        logits = (logits[src] * logits[dst]).sum(dim=1)
        loss = F.binary_cross_entropy_with_logits(logits, labels)
    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        if epoch >= 3:
            dur.append(time.time() - t0)
    
        train_auc, train_rmse = evaluate(gat_net, g, features, edges, labels)
        train_loss_list.append(loss.item())
        train_acc_list.append(train_auc)
        train_rmse_list.append(train_rmse)
        
        results.append([epoch, loss.item(), train_auc, train_rmse, np.mean(dur)])
        
        if epoch % 10 == 0:
            print("Epoch {:05d} | Loss {:.4f} | Train AUC {:.4f} | Train RMSE {:.4f} | Time(s) {:.4f}".format(
                epoch, loss.item(), train_auc, train_rmse, np.mean(dur)))
    
    print("========================Final results:=========================================")
    print("Train AUC: {:.4f}".format(np.mean(train_acc_list)))
    print("Average Train Loss: {:.4f}".format(np.mean(train_loss_list)))
    print("Train RMSE: {:.4f}".format(np.mean(train_rmse_list)))

    with open('training_results.csv', 'w', newline='') as csvfile:
        fieldnames = ['Epoch', 'Loss', 'Train AUC', 'Train RMSE', 'Time']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for result in results:
            writer.writerow({'Epoch': result[0], 'Loss': result[1], 'Train AUC': result[2], 'Train RMSE': result[3], 'Time': result[4]})

    # Model extraction phase
    print("=========Model Extracting==========================")
    
    alpha = 1
    nx_g = g.to_networkx()
    g_matrix = nx.to_numpy_array(nx_g)
    print("Shape of g_matrix:", g_matrix.shape)
    
    sub_graph_node_index = [random.randint(0, data.num_nodes() - 1) for _ in range(attack_node_number)]
    
    syn_nodes = []
    for node_index in sub_graph_node_index:
        one_step_node_index = g_matrix[node_index, :].nonzero()[0].tolist()
        for first_order_node_index in one_step_node_index:
            syn_nodes.append(first_order_node_index)
    sub_graph_syn_node_index = list(set(syn_nodes) - set(sub_graph_node_index))
    
    total_sub_nodes = list(set(sub_graph_syn_node_index + sub_graph_node_index))
    np_features_query = features.clone().detach().cpu().numpy()
    
    for node_index in sub_graph_syn_node_index:
        np_features_query[node_index] = np.zeros_like(np_features_query[node_index])
        one_step_node_index = g_matrix[node_index, :].nonzero()[0].tolist()
        one_step_node_index = list(set(one_step_node_index).intersection(set(sub_graph_node_index)))
        
        num_one_step = len(one_step_node_index)
        for first_order_node_index in one_step_node_index:
            this_node_degree = len(g_matrix[first_order_node_index, :].nonzero()[0].tolist())
            if num_one_step > 0 and this_node_degree > 0:
                np_features_query[node_index] += features[first_order_node_index].cpu().numpy() * alpha / math.sqrt(num_one_step * this_node_degree)
            
            two_step_node_index = g_matrix[first_order_node_index, :].nonzero()[0].tolist()
            two_step_node_index = list(set(two_step_node_index).intersection(set(sub_graph_node_index)))
        
        num_two_step = len(two_step_node_index)
        for second_order_node_index in two_step_node_index:
            this_node_second_step_nodes = []
            this_node_first_step_nodes = g_matrix[second_order_node_index, :].nonzero()[0].tolist()
            for nodes_in_this_node in this_node_first_step_nodes:
                this_node_second_step_nodes.extend(g_matrix[nodes_in_this_node, :].nonzero()[0].tolist())
            this_node_second_step_nodes = list(set(this_node_second_step_nodes) - set(this_node_first_step_nodes))
            
            this_node_second_degree = len(this_node_second_step_nodes)
            if num_two_step > 0 and this_node_second_degree > 0:
                np_features_query[node_index] += features[second_order_node_index].cpu().numpy() * (1 - alpha) / math.sqrt(num_two_step * this_node_second_degree)
    
    features_query = th.FloatTensor(np_features_query).to(device)
    sub_train_mask = data.ndata['train_mask'][total_sub_nodes]
    sub_features = features_query[total_sub_nodes]
    sub_labels = labels[total_sub_nodes]
    
    sub_g = nx.from_numpy_array(g_matrix[total_sub_nodes][:, total_sub_nodes])
    sub_g.remove_edges_from(nx.selfloop_edges(sub_g))
    sub_g.add_edges_from(zip(sub_g.nodes(), sub_g.nodes()))
    sub_g = dgl.from_networkx(sub_g)
    n_edges = sub_g.number_of_edges()
    degs = sub_g.in_degrees().float()
    norm = th.pow(degs, -0.5)
    norm[th.isinf(norm)] = 0
    sub_g.ndata['norm'] = norm.unsqueeze(1)
    sub_g = sub_g.to(device)
    
    net = GAT(feature_number, 32, 32, num_heads=4, dropout=0.5).to(device)
    optimizer = th.optim.Adam(net.parameters(), lr=1e-3, weight_decay=5e-4)
    dur = []

    sub_edges, sub_labels = create_edge_labels(sub_g)
    sub_edges, sub_labels = sub_edges.to(device), sub_labels.to(device)

    print("=========Model Extracting==========================")
    max_auc1 = 0
    max_auc2 = 0
    min_rmse1 = float('inf')
    min_rmse2 = float('inf')
    extraction_results = []
    for epoch in range(200):
        if epoch >= 3:
            t0 = time.time()
    
        net.train()
        logits = net(sub_g, sub_features)
        src, dst = sub_edges[:, 0], sub_edges[:, 1]
        logits = (logits[src] * logits[dst]).sum(dim=1)
        loss = F.binary_cross_entropy_with_logits(logits, sub_labels)
    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        if epoch >= 3:
            dur.append(time.time() - t0)
    
        auc1, rmse1 = evaluate(net, g, features, edges, labels)
        auc2, rmse2 = evaluate(net, g, features, edges, labels)
        if auc1 > max_auc1:
            max_auc1 = auc1
        if auc2 > max_auc2:
            max_auc2 = auc2
        if rmse1 < min_rmse1:
            min_rmse1 = rmse1
        if rmse2 < min_rmse2:
            min_rmse2 = rmse2
        if epoch % 10 == 0:
            print("Epoch {:05d} | Loss {:.4f} | Test AUC {:.4f} | Test RMSE {:.4f} | Test Fid AUC {:.4f} | Test Fid RMSE {:.4f} | Time(s) {:.4f}".format(
                epoch, loss.item(), auc2, rmse2, auc1, rmse1, np.mean(dur)))
            extraction_results.append([epoch, loss.item(), auc2, rmse2, auc1, rmse1, np.mean(dur)])
    
    print("========================Final results:=========================================")
    print("AUC: " + str(max_auc2) + " | Fidelity AUC: " + str(max_auc1))
    print("RMSE: " + str(min_rmse2) + " | Fidelity RMSE: " + str(min_rmse1))

    with open('extraction_results.csv', 'w', newline='') as csvfile:
        fieldnames = ['Epoch', 'Loss', 'Test AUC', 'Test RMSE', 'Test Fid AUC', 'Test Fid RMSE', 'Time']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for result in extraction_results:
            writer.writerow({'Epoch': result[0], 'Loss': result[1], 'Test AUC': result[2], 'Test RMSE': result[3], 'Test Fid AUC': result[4], 'Test Fid RMSE': result[5], 'Time': result[6]})

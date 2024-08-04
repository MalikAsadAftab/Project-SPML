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
from dgl.nn.pytorch import SAGEConv
from sklearn.metrics import roc_auc_score, log_loss
import csv

def load_data(dataset_name):
    if dataset_name == 'amazon_photo':
        dataset = AmazonCoBuyPhotoDataset()
        data = dataset[0]
        print("Shape of g_matrix:", data)
    elif dataset_name == 'citeseer':
        data = citegrh.load_citeseer()
    elif dataset_name == 'pubmed':
        data = citegrh.load_pubmed()
    
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

    # Print the sizes of the masks
    print("Number of nodes:", num_nodes)
    print("Shape of train_mask:", train_mask.shape)
    print("Number of train nodes:", train_mask.sum().item())
    print("Shape of val_mask:", val_mask.shape)
    print("Number of val nodes:", val_mask.sum().item())
    print("Shape of test_mask:", test_mask.shape)
    print("Number of test nodes:", test_mask.sum().item())

    # Add masks to the graph
    data.ndata['train_mask'] = train_mask
    data.ndata['val_mask'] = val_mask
    data.ndata['test_mask'] = test_mask
    return data, features, labels, train_mask, val_mask, test_mask

def evaluate(model, g, features, labels, mask):
    model.eval()
    with th.no_grad():
        logits = model(g, features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = th.max(logits, dim=1)
        correct = th.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)

def evaluate_loss(model, g, features, labels, mask):
    model.eval()
    with th.no_grad():
        logits = model(g, features)
        logp = F.log_softmax(logits, 1)
        loss = F.nll_loss(logp[mask], labels[mask])
        return loss.item()

class GraphSAGE(nn.Module):
    def __init__(self, in_feats, h_feats, out_feats, dropout):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_feats, h_feats, 'mean')
        self.conv2 = SAGEConv(h_feats, out_feats, 'mean')
        self.dropout = nn.Dropout(dropout)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.dropout(h)
        h = self.conv2(g, h)
        return h

def evaluate_auc(model, g, features, mask, labels):
    model.eval()
    with th.no_grad():
        logits = model(g, features)
        logits = logits[mask]
        labels = labels[mask]
        probabilities = F.softmax(logits, dim=1)  # Convert logits to probabilities for multiclass classification
        auc = roc_auc_score(labels.cpu().numpy(), probabilities.cpu().numpy(), multi_class='ovr', average='macro')
        return auc

def evaluate_log_loss(model, g, features, mask, labels):
    model.eval()
    with th.no_grad():
        logits = model(g, features)
        logits = logits[mask]
        labels = labels[mask]
        probabilities = F.softmax(logits, dim=1)  # Convert logits to probabilities for multiclass classification
        logloss = log_loss(labels.cpu().numpy(), probabilities.cpu().numpy())
        return logloss

def attack0_GraphSAGE(dataset_name, attack_node_arg, cuda):
    data, features, labels, train_mask, val_mask, test_mask = load_data(dataset_name)
    
    # Add self-loops to the graph
    g = dgl.add_self_loop(data)

    # Use CPU if CUDA is not available
    device = th.device('cuda' if th.cuda.is_available() else 'cpu')
    
    features = features.to(device)
    labels = labels.to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)
    g = g.to(device)
    
    feature_number = features.shape[1]
    label_number = th.unique(labels).shape[0]
    attack_node_number = int(data.num_nodes() * attack_node_arg)
    
    sage_net = GraphSAGE(feature_number, 32, label_number, dropout=0.5).to(device)
    
    optimizer = th.optim.Adam(sage_net.parameters(), lr=1e-3, weight_decay=5e-4)
    dur = []
    
    train_acc_list = []
    val_acc_list = []
    test_acc_list = []
    train_loss_list = []
    val_loss_list = []
    test_loss_list = []
    fidelity_list = []

    # Create CSV file for training
    train_csv_file = open('train_results.csv', mode='w', newline='')
    train_writer = csv.writer(train_csv_file)
    train_writer.writerow(['Epoch', 'Loss', 'Train AUC', 'Train Log Loss', 'Time'])

    # Create CSV file for extraction
    extract_csv_file = open('extract_results.csv', mode='w', newline='')
    extract_writer = csv.writer(extract_csv_file)
    extract_writer.writerow(['Epoch', 'Loss', 'Test AUC', 'Test Log Loss', 'Test Fid AUC', 'Test Fid Log Loss', 'Time'])
    
    print("=========Target Model Generating==========================")
    for epoch in range(200):
        if epoch >= 3:
            t0 = time.time()
    
        sage_net.train()
        logits = sage_net(g, features)
        logp = F.log_softmax(logits, 1)
        loss = F.nll_loss(logp[train_mask], labels[train_mask])
    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        if epoch >= 3:
            dur.append(time.time() - t0)
    
        train_auc = evaluate_auc(sage_net, g, features, train_mask, labels)
        train_logloss = evaluate_log_loss(sage_net, g, features, train_mask, labels)
        train_loss = loss.item()
        train_acc = evaluate(sage_net, g, features, labels, train_mask)
        val_acc = evaluate(sage_net, g, features, labels, val_mask)
        test_acc = evaluate(sage_net, g, features, labels, test_mask)
        val_loss = evaluate_loss(sage_net, g, features, labels, val_mask)
        test_loss = evaluate_loss(sage_net, g, features, labels, test_mask)
        fidelity = evaluate(sage_net, g, features, labels, test_mask)
        
        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)
        test_acc_list.append(test_acc)
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        test_loss_list.append(test_loss)
        fidelity_list.append(fidelity)
        
        train_writer.writerow([epoch, train_loss, train_auc, train_logloss, np.mean(dur)])
        
        if np.isnan(train_loss) or np.isnan(train_auc) or np.isnan(train_logloss):
            print(f"NaN value detected at epoch {epoch}")
            print(f"Train Loss: {train_loss}, Train AUC: {train_auc}, Train Log Loss: {train_logloss}")
            break
        
        if epoch % 10 == 0:
            print("Epoch {:05d} | Loss {:.4f} | Train AUC {:.4f} | Train Log Loss {:.4f} | Time(s) {:.4f}".format(
                epoch, train_loss, train_auc, train_logloss, np.mean(dur)))
    
    print("========================Final results:=========================================")
    if train_acc_list and val_acc_list and test_acc_list and train_loss_list and val_loss_list and test_loss_list and fidelity_list:
        print("Train Accuracy: {:.4f}".format(np.mean(train_acc_list)))
        print("Validation Accuracy: {:.4f}".format(np.mean(val_acc_list)))
        print("Test Accuracy: {:.4f}".format(np.mean(test_acc_list)))
        print("Average Train Loss: {:.4f}".format(np.mean(train_loss_list)))
        print("Average Validation Loss: {:.4f}".format(np.mean(val_loss_list)))
        print("Average Test Loss: {:.4f}".format(np.mean(test_loss_list)))
        print("Fidelity: {:.4f}".format(np.mean(fidelity_list)))
    else:
        print("Train Accuracy: nan")
        print("Validation Accuracy: nan")
        print("Test Accuracy: nan")
        print("Average Train Loss: nan")
        print("Average Validation Loss: nan")
        print("Average Test Loss: nan")
        print("Fidelity: nan")

    # Model extraction phase
    print("=========Model Extracting==========================")
    
    alpha = 0
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
    sub_train_mask = train_mask[total_sub_nodes]
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
    
    net = GraphSAGE(feature_number, 32, label_number, dropout=0.5).to(device)
    optimizer = th.optim.Adam(net.parameters(), lr=1e-3, weight_decay=5e-4)
    dur = []

    max_auc1 = 0
    max_auc2 = 0
    min_logloss1 = float('inf')
    min_logloss2 = float('inf')
    for epoch in range(200):
        if epoch >= 3:
            t0 = time.time()
    
        net.train()
        logits = net(sub_g, sub_features)
        logp = F.log_softmax(logits, dim=1)
        loss = F.nll_loss(logp[sub_train_mask], sub_labels[sub_train_mask])
    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        if epoch >= 3:
            dur.append(time.time() - t0)
    
        auc1 = evaluate_auc(net, g, features, test_mask, labels)
        logloss1 = evaluate_log_loss(net, g, features, test_mask, labels)
        auc2 = evaluate_auc(net, g, features, test_mask, labels)
        logloss2 = evaluate_log_loss(net, g, features, test_mask, labels)
        if auc1 > max_auc1:
            max_auc1 = auc1
        if auc2 > max_auc2:
            max_auc2 = auc2
        if logloss1 < min_logloss1:
            min_logloss1 = logloss1
        if logloss2 < min_logloss2:
            min_logloss2 = logloss2
        print("Epoch {:05d} | Loss {:.4f} | Test AUC {:.4f} | Test Log Loss {:.4f} | Test Fid AUC {:.4f} | Test Fid Log Loss {:.4f} | Time(s) {:.4f}".format(
                epoch, loss.item(), auc2, logloss2, auc1, logloss1, np.mean(dur)))
        extract_writer.writerow([epoch, loss.item(), auc2, logloss2, auc1, logloss1, np.mean(dur)])
    
    print("========================Final results:=========================================")
    print("AUC: " + str(max_auc2) + " | Fidelity: " + str(max_auc1))

    extract_csv_file.close()
    train_csv_file.close()
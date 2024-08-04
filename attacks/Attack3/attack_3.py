from dgl.data import citation_graph as citegrh
from dgl.data import AmazonCoBuyPhotoDataset
import networkx as nx
import numpy as np
import torch as th
import random
import dgl
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
from dgl.nn.pytorch import GATConv
from sklearn.metrics import roc_auc_score, log_loss
import time
import csv

class GATNet(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, num_heads):
        super(GATNet, self).__init__()
        self.layer1 = GATConv(in_feats, hidden_feats, num_heads=num_heads, allow_zero_in_degree=True)
        self.layer2 = GATConv(hidden_feats * num_heads, out_feats, num_heads=1, allow_zero_in_degree=True)

    def forward(self, g, features):
        x = self.layer1(g, features)
        x = x.flatten(1)
        x = F.elu(x)
        x = self.layer2(g, x)
        x = x.mean(1)
        return x

def evaluate(model, g, features, labels, mask):
    model.eval()
    with th.no_grad():
        logits = model(g, features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = th.max(logits, dim=1)
        correct = th.sum(indices == labels).float()  # Ensure the result is a float tensor
        return correct.item() * 1.0 / len(labels)

def evaluate_loss(model, g, features, labels, mask):
    model.eval()
    with th.no_grad():
        logits = model(g, features)
        logp = F.log_softmax(logits, 1)
        loss = F.nll_loss(logp[mask], labels[mask])
        return loss.item()

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

def attack3(dataset_name, attack_node_arg, cuda):
    data, features, labels, train_mask, val_mask, test_mask = load_data(dataset_name)
    
    node_number = data.number_of_nodes()
    attack_node_number = int(node_number * attack_node_arg)
    
    device = th.device('cuda' if th.cuda.is_available() else 'cpu')
    if cuda:
        data = data.to(device)
        features = features.to(device)
        labels = labels.to(device)
        train_mask = train_mask.to(device)
        test_mask = test_mask.to(device)

    feature_number = features.shape[1]
    label_number = th.unique(labels).shape[0]
    
    # Add self-loops to the graph
    data = dgl.add_self_loop(data)
    
    # Initialize GAT model
    gat_net = GATNet(feature_number, 8, label_number, num_heads=4).to(device)
    
    optimizer = th.optim.Adam(gat_net.parameters(), lr=1e-2, weight_decay=5e-4)
    dur = []

    train_acc_list = []
    val_acc_list = []
    test_acc_list = []
    train_loss_list = []
    val_loss_list = []
    test_loss_list = []
    fidelity_list = []

    print("=========Target Model Generating==========================")
    for epoch in range(200):
        if epoch >= 3:
            t0 = time.time()
    
        gat_net.train()
        logits = gat_net(data, features)
        logp = F.log_softmax(logits, 1)
        loss = F.nll_loss(logp[train_mask], labels[train_mask])
    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        if epoch >= 3:
            dur.append(time.time() - t0)
    
        train_acc = evaluate(gat_net, data, features, labels, train_mask)
        val_acc = evaluate(gat_net, data, features, labels, val_mask)
        test_acc = evaluate(gat_net, data, features, labels, test_mask)
        train_loss = evaluate_loss(gat_net, data, features, labels, train_mask)
        val_loss = evaluate_loss(gat_net, data, features, labels, val_mask)
        test_loss = evaluate_loss(gat_net, data, features, labels, test_mask)
        fidelity = evaluate(gat_net, data, features, labels, test_mask)
        
        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)
        test_acc_list.append(test_acc)
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        test_loss_list.append(test_loss)
        fidelity_list.append(fidelity)
        
        if epoch % 10 == 0:
            print("Epoch {:05d} | Loss {:.4f} | Train Acc {:.4f} | Val Acc {:.4f} | Test Acc {:.4f} | Time(s) {:.4f}".format(
                epoch, loss.item(), train_acc, val_acc, test_acc, np.mean(dur)))
        
    attack_nodes = []
    for i in range(attack_node_number):
        candidate_node = random.randint(0, node_number - 1)
        if candidate_node not in attack_nodes:
            attack_nodes.append(candidate_node)
    
    g_matrix = nx.to_numpy_array(data.cpu().to_networkx())
    features_np = features.cpu().numpy()
    labels_np = labels.cpu().numpy()
    train_mask_np = train_mask.cpu().numpy()
    test_mask_np = test_mask.cpu().numpy()
    
    test_num = 0
    for i in range(node_number):
        if i in attack_nodes:
            test_mask_np[i] = 0
            train_mask_np[i] = 1
            continue
        else:
            if test_num < 1000:
                test_mask_np[i] = 1
                train_mask_np[i] = 0
                test_num += 1
            else:
                test_mask_np[i] = 0
                train_mask_np[i] = 0
    
    gat_net.eval()
    features = th.FloatTensor(features_np).to(device)
    g_graph = dgl.graph((g_matrix.nonzero()[0], g_matrix.nonzero()[1])).to(device)
    
    logits_query = gat_net(g_graph, features)
    _, labels_query = th.max(logits_query, dim=1)
    
    syn_features_np = np.eye(node_number)
    syn_features = th.FloatTensor(syn_features_np).to(device)
    
    g = nx.from_numpy_array(g_matrix)
    g.remove_edges_from(nx.selfloop_edges(g))
    g.add_edges_from(zip(g.nodes(), g.nodes()))
    
    g = dgl.from_networkx(g).to(device)
    n_edges = g.number_of_edges()
    degs = g.in_degrees().float()
    norm = th.pow(degs, -0.5)
    norm[th.isinf(norm)] = 0
    g.ndata['norm'] = norm.unsqueeze(1)
    
    net_attack = GATNet(node_number, 8, label_number, num_heads=4).to(device)
    
    optimizer_original = th.optim.Adam(net_attack.parameters(), lr=5e-2, weight_decay=5e-4)
    dur = []

    max_acc1 = 0
    max_acc2 = 0

    train_data = []
    extraction_data = []

    print("=========Model Extracting==========================")
    
    for epoch in range(200):
        if epoch >= 3:
            t0 = time.time()
            
        net_attack.train()
        logits = net_attack(g, syn_features)
        logp = F.log_softmax(logits, 1)
        loss = F.nll_loss(logp[train_mask], labels_query[train_mask])
        
        optimizer_original.zero_grad()
        loss.backward()
        optimizer_original.step()
        
        if epoch >= 3:
            dur.append(time.time() - t0)
            
        acc1 = evaluate(net_attack, g, syn_features, th.tensor(labels_np).to(device), th.tensor(test_mask_np).to(device))
        acc2 = evaluate(net_attack, g, syn_features, labels_query, th.tensor(test_mask_np).to(device))
        train_auc = evaluate_auc(net_attack, g, syn_features, train_mask, labels_query)
        train_logloss = evaluate_log_loss(net_attack, g, syn_features, train_mask, labels_query)
        test_auc = evaluate_auc(net_attack, g, syn_features, test_mask, labels_query)
        test_logloss = evaluate_log_loss(net_attack, g, syn_features, test_mask, labels_query)
        
        train_data.append([epoch, loss.item(), train_auc, train_logloss, np.mean(dur)])
        extraction_data.append([epoch, loss.item(), acc1, acc2, np.mean(dur)])
        
        if epoch % 10 == 0:
            print("Epoch {:05d} | Loss {:.4f} | Test Acc {:.4f} | Test Fid Acc {:.4f} | Time(s) {:.4f}".format(
                epoch, loss.item(), acc1, acc2, np.mean(dur)))
        
        if acc1 > max_acc1:
            max_acc1 = acc1
        if acc2 > max_acc2:
            max_acc2 = acc2
    
    print("========================Final results:=========================================")
    if train_acc_list and val_acc_list and test_acc_list and train_loss_list and val_loss_list and test_loss_list and fidelity_list:
        print("Avg. Train Accuracy: {:.4f}".format(np.mean(train_acc_list)))
        print("Avg. Validation Accuracy: {:.4f}".format(np.mean(val_acc_list)))
        print("Avg. Test Accuracy: {:.4f}".format(np.mean(test_acc_list)))
        print("Avg. Train Loss: {:.4f}".format(np.mean(train_loss_list)))
        print("Avg. Validation Loss: {:.4f}".format(np.mean(val_loss_list)))
        print("Avg. Test Loss: {:.4f}".format(np.mean(test_loss_list)))
        print("Avg. Fidelity: {:.4f}".format(np.mean(fidelity_list)))
    else:
        print("Avg. Train Accuracy: nan")
        print("Avg. Validation Accuracy: nan")
        print("Avg. Test Accuracy: nan")
        print("Avg. Train Loss: nan")
        print("Avg. Validation Loss: nan")
        print("Avg. Test Loss: nan")
        print("Avg. Fidelity: nan")
    
    # Save to CSV
    with open('train_data.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "Loss", "Train AUC", "Train Log Loss", "Time"])
        writer.writerows(train_data)

    with open('extraction_data.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "Loss", "Test AUC", "Test Log Loss", "Time"])
        writer.writerows(extraction_data)

    print("Max. Accuracy: " + str(max_acc1) + " |  Max. Fidelity: " + str(max_acc2))


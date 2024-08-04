import numpy as np
import torch
import dgl
from dgl.data import AmazonCoBuyPhotoDataset
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GATConv

# Load the AmazonCoBuyPhoto dataset and extract the graph
dataset = AmazonCoBuyPhotoDataset()
graph = dataset[0]

# Convert to NetworkX graph
G = graph.to_networkx()

# Number of nodes in the graph
num_nodes = G.number_of_nodes()
print(f"Number of nodes in the graph: {num_nodes}")

# Extract features from the graph
features = graph.ndata['feat']
num_features = features.shape[0]
print(f"Number of feature vectors: {num_features}")
print(f"Feature vector length: {features.shape[1]}")

# Ensure the number of features matches the number of nodes in the graph
if num_nodes != num_features:
    print(f"Number of nodes ({num_nodes}) and features ({num_features}) do not match.")
    if num_features < num_nodes:
        # Padding feature matrix to match the number of graph nodes
        print("Padding feature matrix to match the number of graph nodes.")
        padding_size = num_nodes - num_features
        padding = torch.zeros(padding_size, features.shape[1])
        features = torch.cat([features, padding], dim=0)
    else:
        raise ValueError(f"Feature matrix has more rows than nodes.")

# Update the graph with the corrected feature matrix
graph.ndata['feat'] = features

# Define the GAT model
class GATLinkPredictor(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, num_heads):
        super(GATLinkPredictor, self).__init__()
        self.layer1 = GATConv(in_feats, hidden_feats, num_heads=num_heads, allow_zero_in_degree=True)
        self.layer2 = GATConv(hidden_feats * num_heads, out_feats, num_heads=1, allow_zero_in_degree=True)

    def forward(self, g, features):
        x = self.layer1(g, features)
        x = F.elu(x)
        x = x.flatten(1)
        x = self.layer2(g, x)
        return x

# Define evaluate_fidelity function
def evaluate_fidelity(baseline_model, attacked_model, graph, features, attack_nodes):
    baseline_model.eval()
    attacked_model.eval()

    with torch.no_grad():
        # Compute outputs for the full graph
        baseline_output = baseline_model(graph, features)
        attacked_output = attacked_model(graph, features)

        # Ensure attack_nodes are within bounds
        if attack_nodes.max().item() >= baseline_output.size(0):
            raise IndexError("Attack node index exceeds the size of the output tensor.")

        # Extract predictions for the attack nodes
        baseline_pred = baseline_output[attack_nodes].squeeze()
        attacked_pred = attacked_output[attack_nodes].squeeze()
        
        # Convert logits to binary predictions if necessary
        baseline_pred = baseline_pred.sigmoid().round()
        attacked_pred = attacked_pred.sigmoid().round()
        
        # Calculate fidelity as the accuracy between the baseline and attacked predictions
        fidelity = (baseline_pred == attacked_pred).float().mean().item()

    return fidelity

# Train baseline model on the original dataset
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
baseline_model = GATLinkPredictor(features.shape[1], 16, 1, num_heads=4).to(device)
baseline_optimizer = torch.optim.Adam(baseline_model.parameters(), lr=1e-2, weight_decay=5e-4)
features = features.to(device)
graph = graph.to(device)

# Generate positive and negative edge samples for training and testing
adj_matrix_dgl = graph.adj().to_dense()
non_edges = (adj_matrix_dgl == 0).nonzero(as_tuple=False)
non_edges = non_edges[non_edges[:, 0] < non_edges[:, 1]]
num_negatives = non_edges.shape[0]
edges = graph.edges()
num_positives = edges[0].shape[0]

negative_samples = non_edges[np.random.choice(num_negatives, num_positives, replace=False)]

edge_labels = torch.cat([torch.ones(num_positives), torch.zeros(len(negative_samples))], dim=0)
all_edges = torch.cat([torch.stack(edges, dim=1), negative_samples], dim=0)

perm = torch.randperm(len(all_edges))
all_edges = all_edges[perm]
edge_labels = edge_labels[perm]

split_idx = int(len(all_edges) * 0.8)
train_edges = all_edges[:split_idx]
train_labels = edge_labels[:split_idx]
test_edges = all_edges[split_idx:]
test_labels = edge_labels[split_idx:]

print("===================Training Baseline Model================================")
for epoch in range(200):
    baseline_model.train()
    baseline_optimizer.zero_grad()

    # Forward pass on the full graph
    baseline_logits = baseline_model(graph, features).squeeze()
    
    # Generate predictions for training edges
    baseline_train_pred = (baseline_logits[train_edges[:, 0]] + baseline_logits[train_edges[:, 1]]) / 2
    baseline_train_pred = baseline_train_pred.sigmoid()  # Assuming binary classification, use sigmoid
    
    # Compute binary cross-entropy loss
    baseline_loss = F.binary_cross_entropy(baseline_train_pred, train_labels.float())

    # Backward pass
    baseline_loss.backward()

    # Update weights
    baseline_optimizer.step()

    baseline_model.eval()
    with torch.no_grad():
        baseline_test_pred = (baseline_logits[test_edges[:, 0]] + baseline_logits[test_edges[:, 1]]) / 2
        baseline_test_pred = baseline_test_pred.sigmoid()
        baseline_test_acc = ((baseline_test_pred > 0.5).float() == test_labels.float()).sum().item() / len(test_labels)

    print(f"Epoch {epoch:05d} | Baseline Loss {baseline_loss.item():.4f} | Baseline Test Acc {baseline_test_acc:.4f}")

print("Baseline model training finished")

# Attack function
def attack1(dataset_name, attack_node_arg, cuda):
    print("==================attack nodes and their queried labels/generated structure loading================================================")
    
    dataset = AmazonCoBuyPhotoDataset()
    data = dataset[0]
    node_number = data.num_nodes()
    feature_number = data.ndata['feat'].shape[1]
    features = data.ndata['feat']

    attack_node_number = int(data.num_nodes() * attack_node_arg)
    print(f"Number of nodes in the graph: {node_number}")
    print(f"Number of attack nodes selected: {attack_node_number}")

    
    # Select attack nodes ensuring indices are within bounds 
    valid_attack_node_indices = np.arange(min(attack_node_number, node_number))
    attack_nodes = np.random.choice(valid_attack_node_indices, attack_node_number, replace=False)
    
    # Extract the subgraph containing the attack nodes
    subgraph = data.subgraph(torch.tensor(attack_nodes, dtype=torch.int64))

    # Ensure the subgraph has nodes and at least one edge
    if subgraph.num_nodes() > 0 and subgraph.num_edges() > 0:
        edges = subgraph.edges()
    else:
        print("Subgraph is empty or has no edges.")
        return

    g_shadow = subgraph
    features_shadow = g_shadow.ndata['feat']

    # Ensure that the number of features matches the number of nodes in the subgraph
    if g_shadow.num_nodes() != features_shadow.shape[0]:
        raise ValueError(f"Number of nodes ({g_shadow.num_nodes()}) and features ({features_shadow.shape[0]}) must match.")

    print(f"Subgraph nodes: {g_shadow.num_nodes()}, Features: {features_shadow.shape[0]}")

    # Generate negative edges for link prediction
    adj_matrix_dgl = g_shadow.adj().to_dense()
    non_edges = (adj_matrix_dgl == 0).nonzero(as_tuple=False)
    non_edges = non_edges[non_edges[:, 0] < non_edges[:, 1]]
    num_negatives = edges[0].shape[0]
    negative_samples = non_edges[np.random.choice(non_edges.shape[0], num_negatives, replace=False)]

    edge_labels = torch.cat([torch.ones(edges[0].shape[0]), torch.zeros(len(negative_samples))], dim=0)

    all_edges = torch.cat([torch.stack(edges, dim=1), negative_samples], dim=0)

    perm = torch.randperm(len(all_edges))
    all_edges = all_edges[perm]
    edge_labels = edge_labels[perm]

    split_idx = int(len(all_edges) * 0.8)
    train_edges = all_edges[:split_idx]
    train_labels = edge_labels[:split_idx]
    test_edges = all_edges[split_idx:]
    test_labels = edge_labels[split_idx:]

    device = torch.device(cuda if cuda and torch.cuda.is_available() else 'cpu')
    model = GATLinkPredictor(feature_number, 16, 1, num_heads=4).to(device)
    features_shadow = features_shadow.to(device)
    g_shadow = g_shadow.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)

    print("===================Model Extracting================================")
    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(g_shadow, features_shadow).squeeze()
        
        # Generate predictions for training edges
        train_pred = (logits[train_edges[:, 0]] + logits[train_edges[:, 1]]) / 2
        train_pred = train_pred.sigmoid()  # Assuming binary classification, use sigmoid to get probabilities
        
        # Compute binary cross-entropy loss
        loss = F.binary_cross_entropy(train_pred, train_labels.float())
        
        # Backward pass
        loss.backward()
        
        # Update weights
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            test_pred = (logits[test_edges[:, 0]] + logits[test_edges[:, 1]]) / 2
            test_pred = test_pred.sigmoid()  # Apply sigmoid to logits for probabilities
            test_acc = ((test_pred > 0.5).float() == test_labels.float()).sum().item() / len(test_labels)

        print(f"Epoch {epoch:05d} | Loss {loss.item():.4f} | Test Acc {test_acc:.4f}")

    print("Training finished")
    
    # Fidelity Evaluation
    baseline_fidelity = evaluate_fidelity(baseline_model, model, g_shadow, features_shadow, torch.tensor(attack_nodes, dtype=torch.long))
    print(f"Fidelity of the attacked model with the baseline model: {baseline_fidelity:.4f}")

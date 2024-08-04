import numpy as np
import torch as th
import dgl
from dgl.data import AmazonCoBuyPhotoDataset
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GATConv
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

# Load the full dataset and set up an oracle model
def setup_oracle(dataset_name):
    if dataset_name == 'amazon_photo':
        dataset = AmazonCoBuyPhotoDataset()
    else:
        raise ValueError("Dataset not supported")
    
    graph = dataset[0]
    features = graph.ndata['feat']
    labels = graph.ndata['label']

    # Create train, validation, and test masks
    num_nodes = graph.num_nodes()
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

    train_mask[train_indices] = True
    val_mask[val_indices] = True
    test_mask[test_indices] = True

    # Assign masks to the graph's node data
    graph.ndata['train_mask'] = train_mask
    graph.ndata['val_mask'] = val_mask
    graph.ndata['test_mask'] = test_mask
    graph.ndata['orig_id'] = th.arange(num_nodes)

    return graph, features, labels


# Simulated Oracle model for querying labels
class OracleModel:
    def __init__(self, labels):
        self.labels = labels
    
    def query_labels(self, nodes):
        # Return the labels for the queried nodes
        return self.labels[nodes]


def query_for_labels(oracle, nodes):
    return oracle.query_labels(nodes)

# LDS graph generation based on node features
def generate_lds_graph(features):
    similarity_matrix = cosine_similarity(features)
    np.fill_diagonal(similarity_matrix, 0)
    # Adjust the threshold to create a more sparse graph
    threshold = np.percentile(similarity_matrix, 90)  # Lower percentile for fewer edges
    adj_matrix = csr_matrix(similarity_matrix)
    adj_matrix.data[adj_matrix.data < threshold] = 0
    adj_matrix.eliminate_zeros()
    return adj_matrix

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

# Train the baseline model
def train_baseline_model(graph, train_mask, features, num_classes, device):
    labels = graph.ndata['label'].to(device)
    model = GATLinkPredictor(features.shape[1], 32, num_classes, num_heads=8).to(device)
    optimizer = th.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    bce_loss = nn.BCEWithLogitsLoss()
    scheduler = th.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    dropout = nn.Dropout(p=0.5)

    for epoch in range(300):
        model.train()
        logits = model(graph, features)  # Shape: [num_nodes, num_classes]
        
        # Apply dropout
        logits = dropout(logits)
        
        # Remove any extra dimensions
        if logits.dim() == 3:  # In case of shape [num_nodes, 1, num_classes]
            logits = logits.squeeze(1)

        labels_one_hot = F.one_hot(labels, num_classes=num_classes).float()
        loss = bce_loss(logits[train_mask], labels_one_hot[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()  # Update the learning rate

        if epoch % 20 == 0:
            print(f"Baseline Epoch {epoch:05d} | Loss {loss.item():.4f}")

    return model

def evaluate_fidelity(attacked_model, baseline_model, graph, features, original_attack_nodes):
    attacked_model.eval()
    baseline_model.eval()

    with th.no_grad():
        # Get predictions from both models
        attacked_output = attacked_model(graph, features)
        baseline_output = baseline_model(graph, features)
        
        # Assuming the outputs are logits, we need to convert them to class indices
        attacked_predictions = attacked_output.argmax(dim=1)
        baseline_predictions = baseline_output.argmax(dim=1)

        # Mapping original attack nodes to the indices used in the shadow graph
        orig_ids = graph.ndata['orig_id']
        node_mapping = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(orig_ids)}
        
        mapped_attack_nodes = []
        unmapped_attack_nodes = []
        
        for node in original_attack_nodes:
            mapped_index = node_mapping.get(node.item(), None)
            if mapped_index is not None:
                mapped_attack_nodes.append(mapped_index)
            else:
                unmapped_attack_nodes.append(node.item())

        if not mapped_attack_nodes:
            print("No attack nodes were mapped correctly.")
            return 0.0

        attack_nodes = th.tensor(mapped_attack_nodes, dtype=th.long)

        # Calculate fidelity
        fidelity = (attacked_predictions[attack_nodes] == baseline_predictions[attack_nodes]).float().mean().item()

        # Debugging outputs
        print(f"Original attack nodes: {original_attack_nodes.tolist()}")
        print(f"Mapped attack nodes: {attack_nodes.tolist()}")
        print(f"Unmapped attack nodes: {unmapped_attack_nodes}")
        print(f"Baseline predictions (Mapped): {baseline_predictions[attack_nodes].tolist()}")
        print(f"Attacked predictions (Mapped): {attacked_predictions[attack_nodes].tolist()}")
        print(f"Number of matching predictions: {(attacked_predictions[attack_nodes] == baseline_predictions[attack_nodes]).sum().item()}")
        print(f"Number of total predictions: {attack_nodes.nelement()}")

    return fidelity

# Check model parameters for differences
def compare_models(baseline_model, attacked_model):
    differences_found = False
    for name, param in baseline_model.named_parameters():
        if not th.equal(param, attacked_model.state_dict()[name]):
            print(f"Difference found in parameter: {name}")
            differences_found = True
            break

    if not differences_found:
        print("No differences found in model parameters.")

# Compare raw logits for differences
def compare_logits(attacked_model, baseline_model, graph, features, original_attack_nodes):
    attacked_model.eval()
    baseline_model.eval()
    
    with th.no_grad():
        attacked_logits = attacked_model(graph, features)
        baseline_logits = baseline_model(graph, features)
        
        # Mapping original attack nodes to the indices used in the shadow graph
        orig_ids = graph.ndata['orig_id']
        node_mapping = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(orig_ids)}
        
        mapped_attack_nodes = []
        unmapped_attack_nodes = []
        
        for node in original_attack_nodes:
            mapped_index = node_mapping.get(node.item(), None)
            if mapped_index is not None and mapped_index < attacked_logits.size(0):
                mapped_attack_nodes.append(mapped_index)
            else:
                unmapped_attack_nodes.append(node.item())

        if not mapped_attack_nodes:
            print("No attack nodes were mapped correctly.")
            return

        attack_nodes = th.tensor(mapped_attack_nodes, dtype=th.long)

        # Compare logits for mapped attack nodes
        attacked_logits = attacked_logits[attack_nodes]
        baseline_logits = baseline_logits[attack_nodes]

        # Check if there's any difference
        differences = (attacked_logits != baseline_logits).sum().item()
        
        print(f"Number of differing logits: {differences}")
        print(f"Attacked logits: {attacked_logits}")
        print(f"Baseline logits: {baseline_logits}")
        print(f"Original attack nodes: {original_attack_nodes.tolist()}")
        print(f"Mapped attack nodes: {attack_nodes.tolist()}")
        print(f"Unmapped attack nodes: {unmapped_attack_nodes}")

# Attack function
def attack2(dataset_name, attack_node_arg, cuda):
    print("================== Starting Attack Scenario ==================")

    # Setup oracle model and data
    graph, features, true_labels = setup_oracle(dataset_name)
    oracle = OracleModel(true_labels)
    num_classes = true_labels.max().item() + 1

    device = th.device('cuda' if cuda else 'cpu')
    
    # Train Baseline Model
    print("================== Training Baseline Model ==================")
    baseline_model = train_baseline_model(graph, graph.ndata['train_mask'], features, num_classes, device)

    attack_node_number = int(graph.num_nodes() * attack_node_arg)
    # Select random attack nodes and query their labels
    print(f"Number of nodes in the graph: {graph.num_nodes()}")
    print(f"Number of attack nodes selected: {attack_node_number}")

    attack_nodes = np.random.choice(graph.num_nodes(), attack_node_number, replace=False)
    queried_labels = query_for_labels(oracle, th.tensor(attack_nodes))

    # Generate LDS graph based on the attributes of attack nodes
    attack_features = features[attack_nodes].numpy()
    adj_matrix = generate_lds_graph(attack_features)
    g_shadow = dgl.from_scipy(adj_matrix)
    attack_features = th.tensor(attack_features, dtype=th.float32)
    g_shadow.ndata['orig_id'] = th.tensor(attack_nodes, dtype=th.int64)

    # Ensure the subgraph has nodes and at least one edge
    if g_shadow.num_nodes() > 0 and g_shadow.num_edges() > 0:
        edges = g_shadow.edges()
    else:
        print("Subgraph is empty or has no edges.")
        return

    print(f"Subgraph nodes: {g_shadow.num_nodes()}, Features: {attack_features.shape[0]}")

    # Generate negative edges for link prediction
    adj_matrix_dgl = g_shadow.adj().to_dense()
    non_edges = (adj_matrix_dgl == 0).nonzero(as_tuple=False)
    non_edges = non_edges[non_edges[:, 0] < non_edges[:, 1]]
    num_negatives = min(edges[0].shape[0], non_edges.shape[0])
    replace_sampling = num_negatives > non_edges.shape[0]

    negative_samples = non_edges[np.random.choice(non_edges.shape[0], num_negatives, replace=replace_sampling)]

    edge_labels = th.cat([th.ones(edges[0].shape[0]), th.zeros(len(negative_samples))], dim=0)

    all_edges = th.cat([th.stack(edges, dim=1), negative_samples], dim=0)

    perm = th.randperm(len(all_edges))
    all_edges = all_edges[perm]
    edge_labels = edge_labels[perm]

    split_idx = int(len(all_edges) * 0.8)
    train_edges = all_edges[:split_idx]
    train_labels = edge_labels[:split_idx]
    test_edges = all_edges[split_idx:]
    test_labels = edge_labels[split_idx:]

    print("================== Training Attacked Model ==================")
    attacked_model = GATLinkPredictor(features.shape[1], 16, 1, num_heads=4).to(device)
    attack_features = attack_features.to(device)
    g_shadow = g_shadow.to(device)
    optimizer = th.optim.Adam(attacked_model.parameters(), lr=1e-2, weight_decay=5e-4)
    for param in attacked_model.parameters():
        param.requires_grad = True

    for epoch in range(200):
        attacked_model.train()
        optimizer.zero_grad()
        
        logits = attacked_model(g_shadow, attack_features).squeeze()
        train_pred = (logits[train_edges[:, 0]] + logits[train_edges[:, 1]]) / 2
        train_pred = train_pred.sigmoid()
        
        loss = F.binary_cross_entropy(train_pred, train_labels.float())
        loss.backward()
        optimizer.step()

        attacked_model.eval()
        with th.no_grad():
            test_pred = (logits[test_edges[:, 0]] + logits[test_edges[:, 1]]) / 2
            test_pred = test_pred.sigmoid()
            test_acc = ((test_pred > 0.5).float() == test_labels.float()).sum().item() / len(test_labels)

        print(f"Attacked Model Epoch {epoch:05d} | Loss {loss.item():.4f} | Test Acc {test_acc:.4f}")

    # Post-training evaluation and comparison
    print("================== Evaluating Fidelity and Test Accuracy ==================")
    fidelity = evaluate_fidelity(attacked_model, baseline_model, g_shadow.to(device), attack_features.to(device), attack_nodes)
    print(f"Fidelity: {fidelity:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print("================== Training and Attack Process Completed ==================")


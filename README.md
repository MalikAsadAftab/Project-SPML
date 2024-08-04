## Overview

This repository implements model extraction attacks on Graph Neural Networks (GNNs), replicating the methodologies described in the paper by Bang Wu, Xiangwen Yang, Shirui Pan, and Xingliang Yuan. The implementation includes several attack scenarios, each simulating different levels of knowledge an attacker might possess.

Here's how you can integrate the provided description into the README file using markdown syntax:

---

## Attack Realization

### Attack-0
We begin with a scenario where the attacker acquires a subset of nodes \(V_A\) from the graph and gains access to their attributes \(X_A\) and their local sub-graph structures \(A_{A,k-hop}\). This reflects realistic scenarios where any node might potentially be an attack node.

To effectively replicate the target model, the attacker constructs what we refer to as an 'attack graph'. This graph incorporates node attributes, graph structure, and node labels derived from adversarial knowledge:

1. **Issuing Queries and Obtaining Labels:** The attacker uses known attributes and the responses from queries to label the attack nodes, employing these labels and attributes to train a duplicate model for node classification.
2. **Gathering Neighbour Connections:** The attacker collects information about connections between the attack nodes and their neighbors to maintain the influence of local graph topology on the node classification accuracy.
3. **Synthesising Attributes for Inaccessible Nodes:** If some neighbor nodes' attributes are unknown, the attacker synthesizes these attributes using a weighted combination of the known attributes of neighboring nodes, adjusted for node connectivity (degree).

These steps culminate in training a node classification GNN model through semi-supervised learning as shown below:

### Attack-1
This attack intensifies the constraints on the attacker's knowledge, limiting it to only the attributes of the attack nodes \(X_A\). In this scenario:

1. **Issuing Queries and Obtaining Labels:** As in Attack-0, attributes and query responses are used to label nodes within the attack graph.
2. **Synthesising Connections Among Attack Nodes:** Without knowledge of the graph structure, the attacker must infer or construct a substitute graph. Using attributes, the attacker employs a graph generation method, such as Learning Discrete Structures (LDS), to synthesize connections among nodes. This method focuses on optimizing the performance of the classification tasks.

After establishing a substitute graph with synthesized connections, the attacker proceeds to train the duplicated model using supervised learning. This approach avoids the isolation of nodes by approximating edge distributions and ensuring a density close to the target graph.

### Attack-2
This attack scenario operates under the constraint that the attacker has access only to the attributes of the attack nodes \(X_A\). The steps involved are:

1. **Issuing Queries and Obtaining Labels:** The attacker selects a set of attack nodes from the graph and uses a query mechanism to obtain labels for these nodes. The selection process ensures that the nodes are within valid bounds, and labels are retrieved using the available features.
2. **Extracting a Subgraph Based on Attack Nodes:** Without the complete graph structure, the attacker focuses on the subset of the graph containing the selected attack nodes. This subgraph is directly extracted from the dataset, ensuring that the number of nodes and the associated features are consistent. The subgraph's connectivity is retained from the original dataset, without additional structure synthesis.
3. **Generating Edge Samples for Training and Testing:** The attacker creates a training set for link prediction by identifying existing edges (positive samples) and generating non-edges (negative samples) within the subgraph. The edge labels are constructed accordingly, and the data is split into training and testing sets.
4. **Training the Attacked Model:** A GAT model is trained on the extracted subgraph. The model training involves optimizing for the binary classification task of predicting whether a pair of nodes is connected. This is done by computing the logits for node pairs and using a binary cross-entropy loss function. The model is updated iteratively, and performance is monitored on the test set.

This approach does not involve generating a synthetic graph structure based on the attributes alone but rather leverages the existing subgraph connectivity from the original data. The attack's effectiveness is gauged by the fidelity metric, which assesses how well the attacked model mimics the baseline model's outputs.

### Attack-3
In this attack scenario, the attacker has full knowledge of the graph structure \(A\) but lacks access to node attributes \(X\) for the vertex set \(V\). The primary challenge is synthesizing node attributes and leveraging the graph structure for conducting a link prediction attack using a GATNet.

#### Steps of the Attack
1. **Graph Structure Utilization:**
   - The attacker utilizes the complete known graph structure \(A\), which includes all connections but no attribute information of any nodes.
2. **Attribute Synthesis for Nodes:**
   - To address the lack of node attribute data, the attacker synthesizes attributes using one-hot encoding based on node indices. For example, for \(n\) nodes, the attribute for node \(v_i\) is a vector of size \(n\) with a 1 at the \(i\)-th position and 0s elsewhere.
   - This synthetic attribute setup ensures each node is uniquely identifiable by its position, which is critical for maintaining structural integrity in the absence of real attribute data.
3. **Model Training Strategy:**
   - The synthesized attributes and the graph structure are used to train a surrogate GATNet model. This model adopts a semi-supervised learning framework where only a subset of node links are labeled.
   - The training focuses on predicting the existence of links between node pairs, leveraging the structural information provided by the synthetic attributes and the complete graph knowledge.
4. **Objective of the Attack:**
   - The attack aims to replicate the target model's ability to predict links, thereby testing the model's robustness against an attack where only structural information is available.
   - This approach also evaluates whether a model trained on synthetic attributes can approximate the predictions of the original model, highlighting potential vulnerabilities in the model's reliance on structure over node-specific attributes.
5. **Challenges and Considerations:**
   - A key challenge is determining whether synthetic attributes can effectively replace real attributes in making meaningful link predictions. This issue is crucial because synthetic attributes may not capture complex attribute-based interactions typical in real scenarios.
   - The computational complexity of training a GATNet on a large graph with high-dimensional one-hot encoded attributes also needs to be considered.


## Generated CSV Files

During the execution of the attacks, various CSV files are generated to log the results and metrics of the training and attack processes. These files include:

### `train_results.csv`
- **Purpose:** Logs metrics such as loss, accuracy, AUC (Area Under the Curve), and other relevant metrics during the training phase of the model under attack.
- **Contents:** The CSV file contains an epoch-by-epoch record of the training process, providing insights into how the model's performance evolves over time.

### `extract_results.csv`
- **Purpose:** Records metrics during the model extraction phase, including test AUC, test log loss, and fidelity scores.
- **Contents:** This file helps in evaluating the success of the attack by comparing the surrogate model's predictions with the target model's behavior. Metrics such as fidelity scores are critical for understanding how well the surrogate model mimics the target model.

### Importance of CSV Files
- **Tracking Training and Evaluation Metrics:** These files provide a structured format for recording key metrics, which are essential for assessing the attack's effectiveness and the model's performance.
- **Detailed Record Keeping:** They allow for epoch-wise tracking, enabling a thorough analysis of the attack process.
- **Post-Attack Analysis:** The recorded data facilitates a deeper understanding of the attack's success and areas for improvement.
- **Reproducibility and Transparency:** Detailed logs ensure that the experiments are reproducible and transparent, adhering to research best practices.

## Running the Code

To run an attack, use the `main.py` script:

```bash
python main.py --attack_type [ATTACK_TYPE] --dataset [DATASET_NAME] --attack_node [ATTACK_NODE_PROPORTION]
```

### Example

```bash
python main.py --attack_type 0 --dataset amazon_photo --attack_node 0.2
```

This command runs the `attack0_GraphSAGE` on the `amazon_photo` dataset with 20% of the nodes used for the attack.

## Reference

Bang Wu, Xiangwen Yang, Shirui Pan, and Xingliang Yuan. "Model Extraction Attacks on Graph Neural Networks: Taxonomy and Realisation". This repository replicates the methodologies and experiments from this paper.

## Acknowledgements

This implementation is inspired by the work of Bang Wu et al., and it aims to provide practical insights into the replication of model extraction attacks on GNNs.

##  How to Use the Additional Information

1. **Overview and Attack Implementation Details:** Provides context and a summary of the various attack strategies implemented in the repository.
2. **Generated CSV Files:** Explains why these files are generated and what they contain, highlighting their importance for monitoring and evaluating the attacks.
3. **Running the Code and Example:** Guides users on how to execute the code and provides an example command.
4. **ProjectSPML-Report.pdf:** This file helps you to understand the details of attacks.
5.  **amazon-books.csv:** This is also a dataset that can be utilized to foresee the attacks impact. The `interaction.csv` and `copurchase_graph.csv` are the supporting files for this dataset.

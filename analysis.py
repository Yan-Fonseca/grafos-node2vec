from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.utils import negative_sampling
import torch
from utils import hadamard
import numpy as np

# from ogb.linkproppred import PygLinkPropPredDataset

# dataset = PygLinkPropPredDataset(name="ogbl-collab")
# data = dataset[0]

# ------------------ CONSTANTS ------------------

DATASET = 'AstroPh_1'
DEVICE = "cuda"

# -----------------------------------------------

embeddings = torch.load(f"embeddings/{DATASET}.pt", weights_only=True)
train_data = torch.load(f"embeddings/{DATASET}_train_data.pt", weights_only=False)
test_data = torch.load(f"embeddings/{DATASET}_test_data.pt", weights_only=False)

neg_edge_index = negative_sampling(
    edge_index=train_data.edge_index,
    num_nodes=train_data.num_nodes,
    num_neg_samples=train_data.edge_label_index.size(1),
)

pos_edge_index = train_data.edge_label_index

edge_index = torch.cat(
    [pos_edge_index, neg_edge_index],
    dim=1
)

labels = torch.cat([
    torch.ones(pos_edge_index.size(1)),
    torch.zeros(neg_edge_index.size(1))
])

z_np = embeddings.numpy()


def hadamard(edges):
    u = edges[0].numpy()
    v = edges[1].numpy()
    return z_np[u] * z_np[v]

x_train = hadamard(edge_index)
y_train = labels.numpy()

x_test = hadamard(test_data.edge_label_index)
y_test = test_data.edge_label.numpy()

clf = LogisticRegression(max_iter=1000)

print(f"x_train: {x_train}")
print(f"y_train: {y_train}")

clf.fit(x_train, y_train)
scores = clf.predict_proba(x_test)[:,1]

print("AUC: ", roc_auc_score(y_test, scores))
print("AP: ", average_precision_score(y_test, scores))

'''
# Get edge splits
split_edge = dataset.get_edge_split()

# Helper function to get edge embeddings
def get_link_prediction_data(edge_index, embeddings, label):
    # Concatenate embeddings of source and target nodes
    # Ensure edge_index is on CPU to index CPU embeddings
    u_emb = embeddings[edge_index[0].cpu()]
    v_emb = embeddings[edge_index[1].cpu()]
    x = torch.cat([u_emb, v_emb], dim=-1)
    y = torch.full((edge_index.size(1),), label, dtype=torch.long)
    return x, y

# Prepare training data
train_pos_edge = split_edge['train']['edge'].to(DEVICE)

# Generate negative samples for training, since 'edge_neg' is not provided for 'train'
num_nodes = data.num_nodes
num_train_pos_edges = train_pos_edge.size(1)

# Transpose train_pos_edge to be [2, num_edges] as expected by negative_sampling
train_neg_edge = negative_sampling(
    edge_index=train_pos_edge.t(),  # Transpose here
    num_nodes=num_nodes,
    num_neg_samples=num_train_pos_edges,
).to(DEVICE)

x_train_pos, y_train_pos = get_link_prediction_data(train_pos_edge.t(), embeddings, 1)
x_train_neg, y_train_neg = get_link_prediction_data(train_neg_edge, embeddings, 0)

x_train = torch.cat([x_train_pos, x_train_neg], dim=0).cpu().numpy()
y_train = torch.cat([y_train_pos, y_train_neg], dim=0).cpu().numpy()

# Prepare validation data
valid_pos_edge = split_edge['valid']['edge'].to(DEVICE)
valid_neg_edge = split_edge['valid']['edge_neg'].to(DEVICE)

x_val_pos, y_val_pos = get_link_prediction_data(valid_pos_edge.t(), embeddings, 1)
x_val_neg, y_val_neg = get_link_prediction_data(valid_neg_edge.t(), embeddings, 0)

x_val = torch.cat([x_val_pos, x_val_neg], dim=0).cpu().numpy()
y_val = torch.cat([y_val_pos, y_val_neg], dim=0).cpu().numpy()

# Prepare test data
test_pos_edge = split_edge['test']['edge'].to(DEVICE)
test_neg_edge = split_edge['test']['edge_neg'].to(DEVICE)

x_test_pos, y_test_pos = get_link_prediction_data(test_pos_edge.t(), embeddings, 1)
x_test_neg, y_test_neg = get_link_prediction_data(test_neg_edge.t(), embeddings, 0)

x_test = torch.cat([x_test_pos, x_test_neg], dim=0).cpu().numpy()
y_test = torch.cat([y_test_pos, y_test_neg], dim=0).cpu().numpy()

print(f"Shape of training features: {x_train.shape}, labels: {y_train.shape}")
print(f"Shape of validation features: {x_val.shape}, labels: {y_val.shape}")
print(f"Shape of test features: {x_test.shape}, labels: {y_test.shape}")

# Train Logistic Regression model
classifier = LogisticRegression(random_state=0, solver='liblinear', C=10, max_iter=1000)
classifier.fit(x_train, y_train)

# Evaluate on validation set
y_pred_val = classifier.predict_proba(x_val)[:, 1]
val_auc = roc_auc_score(y_val, y_pred_val)
print(f"Validation AUC: {val_auc:.4f}")

# Evaluate on test set
y_pred_test = classifier.predict_proba(x_test)[:, 1]
test_auc = roc_auc_score(y_test, y_pred_test)
print(f"Test AUC: {test_auc:.4f}")
'''


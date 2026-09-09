import torch
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

from torch_geometric.utils import negative_sampling


# ============================================================
# CONSTANTS
# ============================================================

DATASET = "Facebook_DeepWalk_1"
GRAPH_DATASET = "facebook_combined"

EMBEDDING_FILE = f"embeddings/{DATASET}.pt"

TRAIN_EDGE_FILE = f"edge_files/{GRAPH_DATASET}_train.edg"
TEST_EDGE_FILE = f"edge_files/{GRAPH_DATASET}_test.edg"

SEED = 42


# ============================================================
# LOAD EMBEDDINGS
# ============================================================

embeddings = torch.load(
    EMBEDDING_FILE,
    weights_only=True
)

embeddings = embeddings.float()

print("Embeddings:", embeddings.shape)


# ============================================================
# LOAD TRAIN EDGES
# ============================================================

train_edges = np.loadtxt(
    TRAIN_EDGE_FILE,
    dtype=np.int64
)

train_edge_index = torch.tensor(
    train_edges.T,
    dtype=torch.long
)


# ============================================================
# LOAD TEST EDGES
# ============================================================

test_edges = np.loadtxt(
    TEST_EDGE_FILE,
    dtype=np.int64
)

test_edge_index = torch.tensor(
    test_edges.T,
    dtype=torch.long
)


print("Train edges:", train_edge_index.shape)
print("Test edges:", test_edge_index.shape)


# ============================================================
# NUMBER OF NODES
# ============================================================

num_nodes = embeddings.size(0)

print("Número de nós:", num_nodes)


# ============================================================
# ORIGINAL GRAPH
# ============================================================
#
# Precisamos garantir que os negativos não sejam:
#
# - arestas de treino
# - arestas de teste
#
# Portanto, usamos todas as arestas do grafo original.
#
# ============================================================

original_edge_index = torch.cat(
    [
        train_edge_index,
        test_edge_index
    ],
    dim=1
)

# Como o grafo é não-direcionado, adicionamos as duas direções.
original_edge_index = torch.cat(
    [
        original_edge_index,
        original_edge_index.flip(0)
    ],
    dim=1
)

# Remove duplicatas
original_edge_index = torch.unique(
    original_edge_index,
    dim=1
)

print("Arestas no grafo original:", original_edge_index.size(1) // 2)


# ============================================================
# NEGATIVE TRAINING EDGES
# ============================================================

torch.manual_seed(SEED)

num_train_positive = train_edge_index.size(1)

train_neg_edge_index = negative_sampling(
    edge_index=original_edge_index,
    num_nodes=num_nodes,
    num_neg_samples=num_train_positive,
    method="sparse",
)


# ============================================================
# NEGATIVE TEST EDGES
# ============================================================

torch.manual_seed(SEED + 1)

num_test_positive = test_edge_index.size(1)

test_neg_edge_index = negative_sampling(
    edge_index=original_edge_index,
    num_nodes=num_nodes,
    num_neg_samples=num_test_positive,
    method="sparse",
)


print("Train positivos:", train_edge_index.size(1))

print("Train negativos:", train_neg_edge_index.size(1))

print("Test positivos:", test_edge_index.size(1))

print("Test negativos:", test_neg_edge_index.size(1))


# ============================================================
# HADAMARD EDGE EMBEDDING
# ============================================================

z = embeddings.numpy()


def hadamard(edge_index):
    u = edge_index[0].numpy()
    v = edge_index[1].numpy()

    return z[u] * z[v]


# ============================================================
# TRAINING DATA
# ============================================================

x_train_pos = hadamard(train_edge_index)
x_train_neg = hadamard(train_neg_edge_index)

x_train = np.concatenate(
    [
        x_train_pos,
        x_train_neg
    ],
    axis=0
)

y_train = np.concatenate(
    [
        np.ones(x_train_pos.shape[0]),
        np.zeros(x_train_neg.shape[0])
    ]
)


# ============================================================
# TEST DATA
# ============================================================

x_test_pos = hadamard(test_edge_index)
x_test_neg = hadamard(test_neg_edge_index)

x_test = np.concatenate(
    [
        x_test_pos,
        x_test_neg
    ],
    axis=0
)

y_test = np.concatenate(
    [
        np.ones(x_test_pos.shape[0]),
        np.zeros(x_test_neg.shape[0])
    ]
)


# ============================================================
# PRINT SHAPES
# ============================================================

print()
print("x_train:", x_train.shape)
print("y_train:", y_train.shape)

print("x_test:", x_test.shape)
print("y_test:", y_test.shape)


# ============================================================
# LOGISTIC REGRESSION
# ============================================================

clf = LogisticRegression(
    max_iter=1000,
    random_state=SEED
)

print()
print("Treinando Logistic Regression...")

clf.fit(
    x_train,
    y_train
)


# ============================================================
# PREDICTION
# ============================================================

scores = clf.predict_proba(
    x_test
)[:, 1]


# ============================================================
# METRICS
# ============================================================

auc = roc_auc_score(
    y_test,
    scores
)

ap = average_precision_score(
    y_test,
    scores
)


print()
print("==============================")
print("RESULTADOS")
print("==============================")

print(f"AUC: {auc:.4f}")
print(f"AP:  {ap:.4f}")

import torch
import numpy as np
from torch_geometric.nn import Node2Vec
from torch_geometric.utils import to_undirected, from_networkx
from torch_geometric.transforms import RandomLinkSplit
import networkx as nx
import random
from torch_cluster import random_walk
from pecanpy import pecanpy
from gensim.models import Word2Vec
import logging
import os
import json

logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s",
    level=logging.INFO,
)

_original_torch_load = torch.load

def patched_load(*args, **kwargs):
	kwargs.setdefault("weights_only", False)
	return _original_torch_load(*args, **kwargs)

torch.load = patched_load

#------------ CONSTANTS -------------------

CURRENT_DIRECTORY = os.getcwd()

DEVICE = "cuda"
CPU = "cpu"

TEST_RATIO = 0.10
SAVE_FILE = "coauth_DBLP_DeepWalk"
DATASET = "coauth-DBLP"

# Node2Vec:

DIM = 128
WALK_LENGTH = 80
CONTEXT_SIZE = 10
WALKS = 10
NEGATIVE_SAMPLES = 1

P = 1.0
Q = 1.0

SPARSE = True

# Model:

BATCH_SIZE = 128
SHUFFLE = True
WORKERS = 4

# Optimizer:

LR = 0.01

EPOCHS = 10

#------------ FUNCTIONS ------------------

def connected_edge_split(data, test_ratio=0.1, seed=42):
    random.seed(seed)

    G = nx.Graph()
    G.add_edges_from(data.edge_index.t().tolist())

    edges = list(G.edges())
    random.shuffle(edges)

    target = int(len(edges) * test_ratio)

    removed = []

    for u, v in edges:
        G.remove_edge(u, v)
        if nx.is_connected(G):
            removed.append((u, v))
        else:
            G.add_edge(u, v)

        if len(removed) >= target:
            break

    train_edges = torch.tensor(
        list(G.edges()),
        dtype=torch.long
    ).t()

    test_edges = torch.tensor(
        removed,
        dtype=torch.long
    ).t()

    return train_edges, test_edges

'''from ogb.linkproppred import PygLinkPropPredDataset

dataset = PygLinkPropPredDataset(name="ogbl-collab")

data = dataset[0]
'''

G = nx.read_edgelist(
    f"dataset/{DATASET}.txt",
    nodetype=int
)

print("Grafo original:")
print("Nós:", G.number_of_nodes())
print("Arestas:", G.number_of_edges())
print("Componentes:", nx.number_connected_components(G))

largest_component = max(
    nx.connected_components(G),
    key=len
)

G = G.subgraph(largest_component).copy()

print()
print("Maior componente conexo:")
print("Nós:", G.number_of_nodes())
print("Arestas:", G.number_of_edges())
print("Conectado:", nx.is_connected(G))

mapping = {
    old_id: new_id
    for new_id, old_id in enumerate(G.nodes())
}

G = nx.relabel_nodes(G, mapping)

print(min(G.nodes()))
print(max(G.nodes()))
print(G.number_of_nodes())


with open(
    f"edge_files/{DATASET}_node_mapping.json",
    "w"
) as f:
    json.dump(mapping, f)

data = from_networkx(G)

#------------------------------------------------
# Separação dos dados de treino dos dados de teste/validação


standard_edge_file_path = CURRENT_DIRECTORY + "/edge_files"
train_file_path = standard_edge_file_path + f"/{DATASET}_train.edg"
test_file_path = standard_edge_file_path + f"/{DATASET}_test.edg"

if (not os.path.exists(train_file_path)) and (not os.path.exists(test_file_path)):
    print("Generating edge files for train and test steps. This might take a while...")

    train_edge_index, test_edge_index = connected_edge_split(
        data,
        test_ratio=0.10,
    )

    train_edge_index = train_edge_index.cpu()
    test_edge_index = test_edge_index.cpu()

    with open(f"edge_files/{DATASET}_train.edg", "w") as f:
        for u, v in train_edge_index.t().tolist():
            f.write(f"{u}\t{v}\n")

    with open(f"edge_files/{DATASET}_test.edg", "w") as f:
        for u, v in test_edge_index.t().tolist():
            f.write(f"{u}\t{v}\n")

    print("Process Done!")

#-------------------------------------------------

# edge_index = train_data.edge_index.to(DEVICE)

# print(train_data)
# print(test_data)

#-------------------------------------------------

g = pecanpy.SparseOTF(
    p=P,
    q=Q,
    workers=WORKERS,
    verbose=True,
)

g.read_edg(
    f"edge_files/{DATASET}_train.edg",
    weighted=False,
    directed=False,
)

walks = g.simulate_walks(
    num_walks=WALKS,
    walk_length=WALK_LENGTH,
)

'''visited = set()

for walk in walks:
    visited.update(map(int, walk))
'''
# print(f"Nós visitados: {len(visited)}")
# print(f"Nós ausentes: {train_data.num_nodes - len(visited)}")

model = Word2Vec(
    sentences=walks,
    vector_size=DIM,
    window=CONTEXT_SIZE,
    min_count=0,
    sg=1,              # Skip-Gram
    workers=WORKERS,
    epochs=EPOCHS,
)

'''
model = Node2Vec(
    edge_index=edge_index,
    embedding_dim=DIM,
    walk_length=WALK_LENGTH,
    context_size=CONTEXT_SIZE,
    walks_per_node=WALKS,
    num_negative_samples=NEGATIVE_SAMPLES,
    p=P,
    q=Q,
    sparse=SPARSE,
    num_nodes=train_data.num_nodes,
).to(DEVICE)

loader = model.loader(
    batch_size=BATCH_SIZE,
    shuffle=SHUFFLE,
    num_workers=WORKERS
)

optimizer = torch.optim.SparseAdam(
    list(model.parameters()),
    lr=LR
)
'''
#---------------------------

num_nodes = len(G.nodes) 
dim = model.vector_size

embeddings = torch.zeros(num_nodes, dim)

for node in range(num_nodes):
        embeddings[node] = torch.tensor(model.wv[str(node)])

#---------------------------
'''
def train():

    model.train()

    total_loss = 0

    for pos_rw, neg_rw in loader:

        optimizer.zero_grad()

        loss = model.loss(
            pos_rw.to(DEVICE),
            neg_rw.to(DEVICE)
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

# -------------------------------------

epochs = EPOCHS

for epoch in range(epochs):

    loss = train()

    print(f"Epoch {epoch+1:03d} | Loss = {loss:.4f}")

model.eval()

with torch.no_grad():
    embeddings = model.embedding.weight.detach().cpu()
'''

print(f"Shape do embedding: {embeddings.shape}")

torch.save(embeddings, f'embeddings/{SAVE_FILE}.pt')
print(f"Embeddings saved to 'embeddings/{SAVE_FILE}.pt'")

# torch.save(train_data, f'embeddings/{SAVE_FILE}_train_data.pt')
# torch.save(test_data, f'embeddings/{SAVE_FILE}_test_data.pt')

# print('Arquivos de treino e test salvos.')

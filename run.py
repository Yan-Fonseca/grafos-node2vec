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

DEVICE = "cuda"
CPU = "cpu"

TEST_RATIO = 0.10
SAVE_FILE = "AstroPh_1"
DATASET = "ca-AstroPh"

# Node2Vec:

DIM = 128
WALK_LENGTH = 80
CONTEXT_SIZE = 10
WALKS = 10
NEGATIVE_SAMPLES = 1

P = 0.25
Q = 0.25

SPARSE = True

# Model:

BATCH_SIZE = 128
SHUFFLE = True
WORKERS = 4

# Optimizer:

LR = 0.01

EPOCHS = 100

#------------ CONSTANTS/ ------------------

'''from ogb.linkproppred import PygLinkPropPredDataset

dataset = PygLinkPropPredDataset(name="ogbl-collab")

data = dataset[0]
'''

G = nx.read_edgelist(
   f"dataset/{DATASET}.txt",
   nodetype=int
)

data = from_networkx(G)

#------------------------------------------------
# Separação dos dados de treino dos dados de teste/validação

transform = RandomLinkSplit(
    num_val=0.0,
    num_test=0.10,
    is_undirected=True,
    add_negative_train_samples=False,
)

train_data, _, test_data = transform(data)

edge_index = train_data.edge_index.cpu()

with open(f"{DATASET}_train.edg", "w") as f:
    for u, v in edge_index.t().tolist():
        f.write(f"{u}\t{v}\n")

#-------------------------------------------------

edge_index = train_data.edge_index.to(DEVICE)

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
    f"{DATASET}_train.edg",
    weighted=False,
    directed=False,
)

walks = g.simulate_walks(
    num_walks=WALKS,
    walk_length=WALK_LENGTH,
)

visited = set()

for walk in walks:
    visited.update(map(int, walk))

print(f"Nós visitados: {len(visited)}")
print(f"Nós ausentes: {train_data.num_nodes - len(visited)}")

missing = set(range(train_data.num_nodes)) - visited
print(sorted(list(missing))[:20])

edge_index = train_data.edge_index

for node in list(missing)[:20]:
    grau = ((edge_index[0] == node) | (edge_index[1] == node)).sum().item()
    print(node, grau)
exit()


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

num_nodes = train_data.num_nodes
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

torch.save(train_data, f'embeddings/{SAVE_FILE}_train_data.pt')
torch.save(test_data, f'embeddings/{SAVE_FILE}_test_data.pt')

print('Arquivos de treino e test salvos.')

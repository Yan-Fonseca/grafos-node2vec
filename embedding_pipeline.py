import os
import pickle
import random
from pathlib import Path

import numpy as np
import torch
from gensim.models import Word2Vec
from pecanpy import pecanpy


SEED = 42
DIM = 128
WALK_LENGTH = 80
CONTEXT_SIZE = 10
WALKS_PER_NODE = 10
P = 1.0
Q = 1.0
WORKERS = 4
EPOCHS = 10


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_fold(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def write_train_edgelist(fold, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for u, v in fold["train_pos"]:
            f.write(f"{u}\t{v}\n")


def generate_fold_embedding(
    fold_path,
    output_path,
    temp_dir="embedding_edge_files",
    dim=DIM,
    walk_length=WALK_LENGTH,
    context_size=CONTEXT_SIZE,
    walks_per_node=WALKS_PER_NODE,
    p=P,
    q=Q,
    workers=WORKERS,
    epochs=EPOCHS,
    seed=SEED,
):
    set_seed(seed)
    fold = load_fold(fold_path)
    fold_idx = int(fold["fold"])
    G_train = fold["G_train"]

    nodes = sorted(G_train.nodes())
    if not nodes:
        raise ValueError(f"Fold {fold_idx} não possui nós.")

    # Os IDs precisam poder indexar diretamente a matriz de embeddings.
    # folds_separation.py preserva os IDs originais, então usamos max_id + 1.
    if not all(isinstance(n, (int, np.integer)) and n >= 0 for n in nodes):
        raise ValueError(
            "Os nós devem possuir IDs inteiros não-negativos. "
            "Se o dataset usa IDs arbitrários, relabele os nós antes de gerar os folds."
        )

    num_nodes = int(max(nodes)) + 1

    temp_edge_path = Path(temp_dir) / f"fold_{fold_idx}_train.edg"
    write_train_edgelist(fold, temp_edge_path)

    walker = pecanpy.SparseOTF(
        p=p,
        q=q,
        workers=workers,
        verbose=True,
    )
    walker.read_edg(
        str(temp_edge_path),
        weighted=False,
        directed=False,
    )

    walks = walker.simulate_walks(
        num_walks=walks_per_node,
        walk_length=walk_length,
    )

    model = Word2Vec(
        sentences=walks,
        vector_size=dim,
        window=context_size,
        min_count=0,
        sg=1,
        workers=workers,
        epochs=epochs,
        seed=seed,
    )

    embeddings = torch.zeros((num_nodes, dim), dtype=torch.float32)
    missing_nodes = []

    # PecanPy/Gensim usa os IDs como strings nos walks.
    for node in nodes:
        key = str(node)
        if key in model.wv:
            embeddings[int(node)] = torch.from_numpy(model.wv[key].copy())
        else:
            missing_nodes.append(int(node))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(embeddings, output_path)

    print(
        f"Fold {fold_idx}: embeddings={tuple(embeddings.shape)} | "
        f"nós sem walk={len(missing_nodes)} | salvo em {output_path}"
    )

    return {
        "fold": fold_idx,
        "embedding_path": str(output_path),
        "num_nodes": num_nodes,
        "missing_nodes": missing_nodes,
    }


def generate_all_embeddings(
    folds_dir,
    embeddings_dir,
    overwrite=False,
    **embedding_kwargs,
):
    folds_dir = Path(folds_dir)
    embeddings_dir = Path(embeddings_dir)
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    fold_paths = sorted(
        folds_dir.glob("fold_*.pkl"),
        key=lambda p: int(p.stem.split("_")[-1]),
    )
    if not fold_paths:
        raise FileNotFoundError(f"Nenhum fold_*.pkl encontrado em {folds_dir}")

    reports = []
    for fold_path in fold_paths:
        fold_idx = int(fold_path.stem.split("_")[-1])
        output_path = embeddings_dir / f"fold_{fold_idx}.pt"

        if output_path.exists() and not overwrite:
            print(f"Fold {fold_idx}: embedding já existe, pulando.")
            reports.append({"fold": fold_idx, "embedding_path": str(output_path)})
            continue

        report = generate_fold_embedding(
            fold_path=fold_path,
            output_path=output_path,
            seed=SEED + fold_idx,
            **embedding_kwargs,
        )
        reports.append(report)

    return reports


if __name__ == "__main__":
    FOLDS_DIR = "/home/souzajbr/grafos/folds_cache-astro-ph"
    EMBEDDINGS_DIR = "/home/souzajbr/grafos/embeddings/astro-ph"

    generate_all_embeddings(
        folds_dir=FOLDS_DIR,
        embeddings_dir=EMBEDDINGS_DIR,
        overwrite=False,
    )

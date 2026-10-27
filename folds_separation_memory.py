"""Memory-efficient K-fold split for large undirected link-prediction graphs.

Differences from the original implementation:
- processes/saves one fold at a time (does not retain all folds in RAM);
- saves compact NumPy arrays (.npz), never a NetworkX G_train per fold;
- does not enforce negative-edge uniqueness across different folds;
- uses integer-encoded negative pairs instead of frozenset objects;
- avoids dtype=object and Python lists for millions of edges;
- when preserve_connectivity=True, builds a spanning forest once and keeps
  those edges in every training fold.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


def load_graph(edge_list_path: str, sep=r"\s+") -> nx.Graph:
    df = pd.read_csv(
        edge_list_path,
        sep=sep,
        header=None,
        names=["source", "target"],
        engine="c",
        dtype=np.int64,
    )
    G = nx.from_pandas_edgelist(df, source="source", target="target")
    del df
    return G


def _choose_int_dtype(max_node_id: int):
    return np.int32 if max_node_id <= np.iinfo(np.int32).max else np.int64


def _edge_key(u: int, v: int, base: int) -> int:
    if u > v:
        u, v = v, u
    return int(u) * base + int(v)


def get_spanning_forest_keys(G: nx.Graph, base: int) -> set[int]:
    """Returns encoded edges of one BFS spanning tree per component."""
    visited = set()
    protected = set()
    for root in G.nodes:
        if root in visited:
            continue
        visited.add(root)
        for u, v in nx.bfs_edges(G, root):
            visited.add(v)
            protected.add(_edge_key(int(u), int(v), base))
    return protected


def graph_edges_to_array(G: nx.Graph, dtype, protected_keys=None, base=None, keep_protected=False):
    """Materialize either protected or unprotected graph edges as a compact array."""
    if protected_keys is None:
        arr = np.empty((G.number_of_edges(), 2), dtype=dtype)
        for i, (u, v) in enumerate(G.edges()):
            arr[i] = (u, v)
        return arr

    count = len(protected_keys) if keep_protected else G.number_of_edges() - len(protected_keys)
    arr = np.empty((count, 2), dtype=dtype)
    j = 0
    for u, v in G.edges():
        is_protected = _edge_key(int(u), int(v), base) in protected_keys
        if is_protected == keep_protected:
            arr[j] = (u, v)
            j += 1
    if j != count:
        arr = arr[:j]
    return arr


def negative_sampling_compact(
    G: nx.Graph,
    n_samples: int,
    *,
    seed: int,
    base: int,
    dtype,
    exclude_keys: set[int] | None = None,
    batch_size: int = 250_000,
) -> tuple[np.ndarray, set[int]]:
    """Samples unique non-edges while keeping only encoded integer keys in a set."""
    rng = np.random.default_rng(seed)
    nodes = np.asarray(list(G.nodes()), dtype=dtype)
    selected_keys = set() if exclude_keys is None else exclude_keys
    initial_count = len(selected_keys)
    out = np.empty((n_samples, 2), dtype=dtype)
    written = 0

    # Sparse collaboration graphs have very high acceptance, so batched proposals
    # reduce Python RNG overhead while G.has_edge performs the actual membership test.
    while written < n_samples:
        need = n_samples - written
        proposals = min(max(need * 2, 10_000), batch_size)
        us = rng.choice(nodes, size=proposals, replace=True)
        vs = rng.choice(nodes, size=proposals, replace=True)

        for u_raw, v_raw in zip(us, vs):
            if written >= n_samples:
                break
            u, v = int(u_raw), int(v_raw)
            if u == v:
                continue
            if u > v:
                u, v = v, u
            key = u * base + v
            if key in selected_keys or G.has_edge(u, v):
                continue
            selected_keys.add(key)
            out[written] = (u, v)
            written += 1

    new_keys = selected_keys if exclude_keys is None else selected_keys
    return out, new_keys


def save_fold_npz(path: Path, fold_idx: int, train_pos, train_neg, test_pos, test_neg):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        fold=np.asarray([fold_idx], dtype=np.int16),
        train_pos=np.ascontiguousarray(train_pos),
        train_neg=np.ascontiguousarray(train_neg),
        test_pos=np.ascontiguousarray(test_pos),
        test_neg=np.ascontiguousarray(test_neg),
    )
    print(f"          -> fold compacto salvo em '{path}' ({path.stat().st_size / 1024**2:.1f} MB)")


def load_fold_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {
            "fold": int(z["fold"][0]),
            "train_pos": z["train_pos"],
            "train_neg": z["train_neg"],
            "test_pos": z["test_pos"],
            "test_neg": z["test_neg"],
        }


def kfold_link_prediction_split_memory_efficient(
    edge_list_path,
    n_splits=5,
    neg_ratio=1.0,
    random_state=42,
    preserve_connectivity=True,
    save_dir="folds_compact",
    resume=True,
):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("Carregando grafo completo...")
    G = load_graph(edge_list_path)
    if G.number_of_nodes() == 0:
        raise ValueError("Grafo vazio.")
    if any(int(n) < 0 for n in G.nodes()):
        raise ValueError("IDs de nós devem ser inteiros não-negativos.")

    max_node = int(max(G.nodes()))
    base = max_node + 1
    dtype = _choose_int_dtype(max_node)

    print(f"Nós={G.number_of_nodes():,} | arestas={G.number_of_edges():,} | dtype={np.dtype(dtype)}")

    if preserve_connectivity:
        print("Construindo floresta geradora compacta...")
        protected_keys = get_spanning_forest_keys(G, base)
        protected_edges = graph_edges_to_array(
            G, dtype=dtype, protected_keys=protected_keys, base=base, keep_protected=True
        )
        candidate_edges = graph_edges_to_array(
            G, dtype=dtype, protected_keys=protected_keys, base=base, keep_protected=False
        )
        print(
            f"[preserve_connectivity=True] {len(protected_edges):,} protegidas; "
            f"{len(candidate_edges):,} elegíveis para teste."
        )
        del protected_keys
    else:
        protected_edges = np.empty((0, 2), dtype=dtype)
        candidate_edges = graph_edges_to_array(G, dtype=dtype)

    meta = {
        "edge_list_path": str(edge_list_path),
        "n_splits": int(n_splits),
        "neg_ratio": float(neg_ratio),
        "random_state": int(random_state),
        "preserve_connectivity": bool(preserve_connectivity),
        "num_nodes": int(G.number_of_nodes()),
        "num_edges": int(G.number_of_edges()),
        "max_node_id": max_node,
        "dtype": str(np.dtype(dtype)),
    }
    with (save_dir / "metadata.json").open("w") as f:
        json.dump(meta, f, indent=2)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(candidate_edges)):
        fold_path = save_dir / f"fold_{fold_idx}.npz"
        if resume and fold_path.exists():
            print(f"Fold {fold_idx}: já existe em '{fold_path}', pulando sem carregá-lo na RAM.")
            continue

        print(f"Fold {fold_idx}: preparando positivos...")
        test_pos = np.ascontiguousarray(candidate_edges[test_idx])
        cand_train = candidate_edges[train_idx]
        train_pos = np.empty((len(protected_edges) + len(cand_train), 2), dtype=dtype)
        train_pos[: len(protected_edges)] = protected_edges
        train_pos[len(protected_edges):] = cand_train
        del cand_train, train_idx, test_idx

        n_train_neg = int(len(train_pos) * neg_ratio)
        n_test_neg = int(len(test_pos) * neg_ratio)

        print(f"Fold {fold_idx}: amostrando {n_train_neg:,} negativos de treino...")
        train_neg, used_in_fold = negative_sampling_compact(
            G, n_train_neg, seed=random_state + fold_idx, base=base, dtype=dtype
        )
        print(f"Fold {fold_idx}: amostrando {n_test_neg:,} negativos de teste...")
        test_neg, _ = negative_sampling_compact(
            G,
            n_test_neg,
            seed=random_state + 10_000 + fold_idx,
            base=base,
            dtype=dtype,
            exclude_keys=used_in_fold,
        )
        del used_in_fold

        save_fold_npz(fold_path, fold_idx, train_pos, train_neg, test_pos, test_neg)
        print(
            f"Fold {fold_idx}: train_pos={len(train_pos):,}, train_neg={len(train_neg):,}, "
            f"test_pos={len(test_pos):,}, test_neg={len(test_neg):,}"
        )
        del train_pos, train_neg, test_pos, test_neg

    print("Todos os folds disponíveis em formato compacto.")
    return str(save_dir)


if __name__ == "__main__":
    DATASET = "coauth-DBLP"
    kfold_link_prediction_split_memory_efficient(
        edge_list_path=f"/home/souzajbr/grafos/dataset/{DATASET}.txt",
        n_splits=5,
        neg_ratio=1.0,
        random_state=42,
        preserve_connectivity=True,
        save_dir=f"/home/souzajbr/grafos/folds_compact-{DATASET}",
        resume=True,
    )

"""Memory-efficient DeepWalk embedding for compact folds.

For P=Q=1 (the configuration used in the original run.py), random walks are
written to disk in small batches and Gensim trains from corpus_file. This avoids
holding every walk in RAM. For P/Q != 1, this module intentionally raises an
error rather than silently falling back to the memory-heavy PecanPy API.
"""

from __future__ import annotations

import gc
import random
from pathlib import Path

import numpy as np
import torch
from gensim.models import Word2Vec
from numba import njit
from scipy.sparse import csr_matrix

from folds_separation_memory import load_fold_npz

SEED = 42
DIM = 128
WALK_LENGTH = 80
CONTEXT_SIZE = 10
WALKS_PER_NODE = 10
P = 1.0
Q = 1.0
WORKERS = 4
EPOCHS = 10
WALK_BATCH_SIZE = 20_000


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@njit(cache=True)
def _walk_batch(indptr, indices, starts, walk_length, seed):
    """Generate first-order random walks (DeepWalk; equivalent to p=q=1)."""
    np.random.seed(seed)
    out = np.empty((len(starts), walk_length), dtype=np.int64)
    lengths = np.ones(len(starts), dtype=np.int32)
    for i in range(len(starts)):
        cur = int(starts[i])
        out[i, 0] = cur
        length = 1
        for j in range(1, walk_length):
            lo = indptr[cur]
            hi = indptr[cur + 1]
            if hi <= lo:
                break
            cur = int(indices[lo + np.random.randint(hi - lo)])
            out[i, j] = cur
            length += 1
        lengths[i] = length
    return out, lengths


def build_csr(train_pos: np.ndarray, num_nodes: int):
    u = train_pos[:, 0].astype(np.int64, copy=False)
    v = train_pos[:, 1].astype(np.int64, copy=False)
    rows = np.concatenate((u, v))
    cols = np.concatenate((v, u))
    data = np.ones(len(rows), dtype=np.uint8)
    A = csr_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes), dtype=np.uint8)
    A.sum_duplicates()
    A.sort_indices()
    del rows, cols, data
    return A


def write_walk_corpus_streaming(
    train_pos: np.ndarray,
    corpus_path: Path,
    *,
    num_nodes: int,
    num_walks: int,
    walk_length: int,
    batch_size: int,
    seed: int,
):
    """Generate a text corpus without ever retaining the full walk set in RAM."""
    print("  construindo CSR para DeepWalk...")
    A = build_csr(train_pos, num_nodes)
    indptr = A.indptr.astype(np.int64, copy=False)
    indices = A.indices.astype(np.int64, copy=False)
    active_nodes = np.flatnonzero(np.diff(indptr) > 0).astype(np.int64)
    print(f"  nós com grau > 0: {len(active_nodes):,}/{num_nodes:,}")

    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    total_walks = len(active_nodes) * num_walks
    done = 0

    with corpus_path.open("w", buffering=8 * 1024 * 1024) as f:
        for walk_round in range(num_walks):
            order = active_nodes.copy()
            rng.shuffle(order)
            for start in range(0, len(order), batch_size):
                starts = order[start:start + batch_size]
                walks, lengths = _walk_batch(
                    indptr, indices, starts, walk_length,
                    seed + walk_round * 1_000_003 + start,
                )
                # Most nodes in a connected training graph have full-length walks.
                # Write by row so early-stopped walks do not contain uninitialized IDs.
                for row, ln in zip(walks, lengths):
                    f.write(" ".join(map(str, row[:int(ln)])))
                    f.write("\n")
                done += len(starts)
                if done % max(batch_size * 10, 1) == 0 or done == total_walks:
                    print(f"  walks: {done:,}/{total_walks:,} ({100 * done / total_walks:.1f}%)")
                del walks, lengths
            del order

    del A, indptr, indices, active_nodes
    gc.collect()


def train_word2vec_from_corpus(
    corpus_path: Path,
    *,
    dim: int,
    context_size: int,
    workers: int,
    epochs: int,
    seed: int,
):
    print("  construindo vocabulário Word2Vec a partir do corpus em disco...")
    model = Word2Vec(
        vector_size=dim,
        window=context_size,
        min_count=0,
        sg=1,
        workers=workers,
        seed=seed,
    )
    model.build_vocab(corpus_file=str(corpus_path))
    print(
        f"  vocabulário={len(model.wv):,} | "
        f"palavras no corpus={model.corpus_total_words:,}"
    )
    model.train(
        corpus_file=str(corpus_path),
        total_words=model.corpus_total_words,
        epochs=epochs,
    )
    return model


def generate_fold_embedding(
    fold_path,
    output_path,
    temp_dir,
    dim=DIM,
    walk_length=WALK_LENGTH,
    context_size=CONTEXT_SIZE,
    walks_per_node=WALKS_PER_NODE,
    p=P,
    q=Q,
    workers=WORKERS,
    epochs=EPOCHS,
    walk_batch_size=WALK_BATCH_SIZE,
    seed=SEED,
):
    if p != 1.0 or q != 1.0:
        raise ValueError(
            "A versão memory-efficient implementa o caso p=q=1 (DeepWalk), "
            "que é exatamente a configuração do run.py original. Para Node2Vec "
            "com p/q diferentes de 1 é necessário um gerador de caminhadas "
            "de segunda ordem também streaming."
        )

    set_seed(seed)
    fold = load_fold_npz(fold_path)
    idx = fold["fold"]
    train_pos = fold["train_pos"]
    num_nodes = int(max(
        train_pos.max(initial=0),
        fold["test_pos"].max(initial=0),
        fold["train_neg"].max(initial=0),
        fold["test_neg"].max(initial=0),
    )) + 1

    # Negatives/test are not needed for embedding; release references now.
    del fold
    gc.collect()

    corpus_path = Path(temp_dir) / f"fold_{idx}_walks.txt"
    print(f"Fold {idx}: gerando corpus DeepWalk streaming em {corpus_path}")
    write_walk_corpus_streaming(
        train_pos,
        corpus_path,
        num_nodes=num_nodes,
        num_walks=walks_per_node,
        walk_length=walk_length,
        batch_size=walk_batch_size,
        seed=seed,
    )
    del train_pos
    gc.collect()

    model = train_word2vec_from_corpus(
        corpus_path,
        dim=dim,
        context_size=context_size,
        workers=workers,
        epochs=epochs,
        seed=seed,
    )

    embeddings = np.zeros((num_nodes, dim), dtype=np.float32)
    missing = 0
    for node in range(num_nodes):
        key = str(node)
        if key in model.wv:
            embeddings[node] = model.wv[key]
        else:
            missing += 1

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(torch.from_numpy(embeddings), output_path)
    print(f"Fold {idx}: embedding={embeddings.shape}, ausentes={missing:,}, salvo em {output_path}")

    del model, embeddings
    gc.collect()
    try:
        corpus_path.unlink()
    except FileNotFoundError:
        pass


def generate_all_embeddings(folds_dir, embeddings_dir, temp_dir, overwrite=False, **kwargs):
    folds_dir = Path(folds_dir)
    embeddings_dir = Path(embeddings_dir)
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    fold_paths = sorted(folds_dir.glob("fold_*.npz"), key=lambda p: int(p.stem.split("_")[-1]))
    if not fold_paths:
        raise FileNotFoundError(f"Nenhum fold_*.npz em {folds_dir}")

    for fold_path in fold_paths:
        idx = int(fold_path.stem.split("_")[-1])
        out = embeddings_dir / f"fold_{idx}.pt"
        if out.exists() and not overwrite:
            print(f"Fold {idx}: embedding já existe, pulando.")
            continue
        generate_fold_embedding(fold_path, out, temp_dir=temp_dir, seed=SEED + idx, **kwargs)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

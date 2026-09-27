"""Memory-aware classifier evaluation for very large link-prediction folds.

Training uses the same reproducible stratified edge sample for all classifiers
when a fold exceeds MAX_TRAIN_PER_CLASS. Test predictions are generated in
batches, so the full Hadamard X_test is never materialized.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix, f1_score, fbeta_score, log_loss,
    matthews_corrcoef, precision_score, recall_score, roc_auc_score,
)
from xgboost import XGBClassifier

from folds_separation_memory import load_fold_npz

SEED = 42
THRESHOLD = 0.5
MAX_TRAIN_PER_CLASS = 500_000
FEATURE_BATCH_SIZE = 200_000


def load_embeddings(path):
    x = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float32)


def hadamard_batch(embeddings, edges):
    edges = np.asarray(edges)
    u = edges[:, 0].astype(np.int64, copy=False)
    v = edges[:, 1].astype(np.int64, copy=False)
    if len(edges) and max(int(u.max()), int(v.max())) >= embeddings.shape[0]:
        raise IndexError("Uma aresta referencia ID fora da matriz de embeddings.")
    return np.multiply(embeddings[u], embeddings[v], dtype=np.float32)


def sample_edges(edges, max_samples, rng):
    if max_samples is None or len(edges) <= max_samples:
        return edges
    idx = rng.choice(len(edges), size=max_samples, replace=False)
    return edges[idx]


def build_training_matrix(fold, embeddings, max_per_class, seed):
    rng = np.random.default_rng(seed)
    pos = sample_edges(fold["train_pos"], max_per_class, rng)
    neg = sample_edges(fold["train_neg"], max_per_class, rng)

    xp = hadamard_batch(embeddings, pos)
    xn = hadamard_batch(embeddings, neg)
    X = np.empty((len(xp) + len(xn), embeddings.shape[1]), dtype=np.float32)
    X[:len(xp)] = xp
    X[len(xp):] = xn
    y = np.empty(len(X), dtype=np.uint8)
    y[:len(xp)] = 1
    y[len(xp):] = 0
    del xp, xn
    return X, y, len(pos), len(neg)


def standardize_inplace(X, mean, scale):
    X -= mean
    X /= scale
    return X


def fit_models(X_train, y_train, seed):
    # Manual scaling prevents sklearn Pipeline/StandardScaler from creating
    # another large full-size training matrix internally.
    mean = X_train.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = X_train.std(axis=0, dtype=np.float64).astype(np.float32)
    scale[scale == 0] = 1.0

    models = {}
    train_times = {}

    print("  treinando logistic_regression...")
    X_lr = X_train.copy()
    standardize_inplace(X_lr, mean, scale)
    lr = LogisticRegression(max_iter=1000, random_state=seed, solver="lbfgs")
    t0 = perf_counter()
    lr.fit(X_lr, y_train)
    train_times["logistic_regression"] = perf_counter() - t0
    print(f"    tempo de treino: {train_times['logistic_regression']:.2f} s")
    del X_lr
    gc.collect()
    models["logistic_regression"] = (lr, True)

    print("  treinando random_forest...")
    rf = RandomForestClassifier(
        n_estimators=250,
        max_features="sqrt",
        max_depth=24,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=seed,
        n_jobs=-1,
    )
    t0 = perf_counter()
    rf.fit(X_train, y_train)
    train_times["random_forest"] = perf_counter() - t0
    print(f"    tempo de treino: {train_times['random_forest']:.2f} s")
    models["random_forest"] = (rf, False)

    print("  treinando xgboost...")
    xgb = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=400,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=-1,
        tree_method="hist",
        max_bin=256,
    )
    t0 = perf_counter()
    xgb.fit(X_train, y_train)
    train_times["xgboost"] = perf_counter() - t0
    print(f"    tempo de treino: {train_times['xgboost']:.2f} s")
    models["xgboost"] = (xgb, False)
    return models, mean, scale, train_times


def predict_edges_batched(model, needs_scaling, embeddings, edges, mean, scale, batch_size):
    out = np.empty(len(edges), dtype=np.float32)
    for start in range(0, len(edges), batch_size):
        end = min(start + batch_size, len(edges))
        X = hadamard_batch(embeddings, edges[start:end])
        if needs_scaling:
            standardize_inplace(X, mean, scale)
        out[start:end] = model.predict_proba(X)[:, 1].astype(np.float32, copy=False)
        del X
    return out


def compute_metrics(y_true, probabilities, threshold):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    pred = (probabilities >= threshold).astype(np.uint8)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "specificity": tn / (tn + fp) if tn + fp else 0.0,
        "f1": f1_score(y_true, pred, zero_division=0),
        "f0.5": fbeta_score(y_true, pred, beta=0.5, zero_division=0),
        "f2": fbeta_score(y_true, pred, beta=2.0, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "average_precision": average_precision_score(y_true, probabilities),
        "mcc": matthews_corrcoef(y_true, pred),
        "log_loss": log_loss(y_true, probabilities, labels=[0, 1]),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def evaluate_fold(
    fold_path, embedding_path, *, threshold=THRESHOLD,
    max_train_per_class=MAX_TRAIN_PER_CLASS,
    feature_batch_size=FEATURE_BATCH_SIZE,
    ensemble_weights=None, seed=SEED,
):
    fold = load_fold_npz(fold_path)
    idx = fold["fold"]
    embeddings = load_embeddings(embedding_path)

    print(f"Fold {idx}: construindo amostra de treino (máx. {max_train_per_class:,}/classe)...")
    X_train, y_train, npos, nneg = build_training_matrix(
        fold, embeddings, max_train_per_class, seed + idx
    )
    print(f"Fold {idx}: X_train={X_train.shape}, {X_train.nbytes / 1024**3:.2f} GiB")

    models, mean, scale, train_times = fit_models(X_train, y_train, seed + idx)
    del X_train, y_train
    gc.collect()

    test_pos, test_neg = fold["test_pos"], fold["test_neg"]
    y_test = np.concatenate((
        np.ones(len(test_pos), dtype=np.uint8),
        np.zeros(len(test_neg), dtype=np.uint8),
    ))

    probs = {}
    prediction_times = {}
    rows = []
    for name, (model, scaled) in models.items():
        print(f"Fold {idx}: predizendo {name} em lotes...")
        t0 = perf_counter()
        pp = predict_edges_batched(model, scaled, embeddings, test_pos, mean, scale, feature_batch_size)
        pn = predict_edges_batched(model, scaled, embeddings, test_neg, mean, scale, feature_batch_size)
        prediction_times[name] = perf_counter() - t0
        print(f"    tempo de predição: {prediction_times[name]:.2f} s")

        p = np.concatenate((pp, pn))
        probs[name] = p
        row = compute_metrics(y_test, p, threshold)
        row.update({
            "fold": idx,
            "model": name,
            "train_time_seconds": float(train_times[name]),
            "prediction_time_seconds": float(prediction_times[name]),
            "combination_time_seconds": 0.0,
            "end_to_end_time_seconds": float(train_times[name] + prediction_times[name]),
        })
        rows.append(row)
        del pp, pn

    names = list(probs)
    committee_base_time = float(sum(train_times.values()) + sum(prediction_times.values()))

    # ========================================================
    # 1) HARD VOTING
    # ========================================================
    # Cada classificador fornece um voto binário. Com 3 modelos,
    # a classe positiva vence quando pelo menos 2 votam em 1.
    t0 = perf_counter()
    hard_votes = np.zeros(len(y_test), dtype=np.uint8)
    for name in names:
        hard_votes += (probs[name] >= threshold).astype(np.uint8)

    hard_pred = (hard_votes >= (len(names) // 2 + 1)).astype(np.uint8)

    # compute_metrics recebe scores contínuos para AUC/AP/log-loss.
    # Para hard voting não há uma probabilidade propriamente dita.
    # Usamos a fração de votos positivos como score discreto (0, 1/3,
    # 2/3, 1), preservando a decisão majoritária no threshold=0.5.
    hard_score = hard_votes.astype(np.float32) / np.float32(len(names))
    hard_combination_time = perf_counter() - t0
    row = compute_metrics(y_test, hard_score, threshold)
    row.update({
        "fold": idx,
        "model": "committee_hard_voting",
        "train_time_seconds": 0.0,
        "prediction_time_seconds": 0.0,
        "combination_time_seconds": float(hard_combination_time),
        "end_to_end_time_seconds": float(committee_base_time + hard_combination_time),
    })
    rows.append(row)

    # ========================================================
    # 2) SOFT VOTING NÃO PONDERADO
    # ========================================================
    # Média aritmética das probabilidades dos classificadores.
    t0 = perf_counter()
    soft_score = np.zeros(len(y_test), dtype=np.float32)
    for name in names:
        soft_score += probs[name]
    soft_score /= np.float32(len(names))
    soft_combination_time = perf_counter() - t0

    row = compute_metrics(y_test, soft_score, threshold)
    row.update({
        "fold": idx,
        "model": "committee_soft_voting",
        "train_time_seconds": 0.0,
        "prediction_time_seconds": 0.0,
        "combination_time_seconds": float(soft_combination_time),
        "end_to_end_time_seconds": float(committee_base_time + soft_combination_time),
    })
    rows.append(row)

    # ========================================================
    # 3) SOFT VOTING PONDERADO
    # ========================================================
    # Os pesos devem ser definidos previamente ou obtidos somente a
    # partir de dados de treino/validação. Nunca escolha pesos olhando
    # o desempenho no conjunto de teste.
    if ensemble_weights is None:
        # Sem pesos fornecidos, mantém pesos iguais. Nesse caso o resultado
        # será numericamente igual ao soft voting tradicional, mas a linha
        # é mantida para deixar o experimento explícito.
        w = np.ones(len(names), dtype=np.float64)
    else:
        missing = [name for name in names if name not in ensemble_weights]
        if missing:
            raise ValueError(
                "Faltam pesos para os classificadores: " + ", ".join(missing)
            )
        w = np.asarray([ensemble_weights[name] for name in names], dtype=np.float64)
        if not np.all(np.isfinite(w)) or (w < 0).any() or w.sum() <= 0:
            raise ValueError("Pesos inválidos para o comitê ponderado.")

    w /= w.sum()

    t0 = perf_counter()
    weighted_score = np.zeros(len(y_test), dtype=np.float32)
    for wi, name in zip(w, names):
        weighted_score += np.float32(wi) * probs[name]
    weighted_combination_time = perf_counter() - t0

    row = compute_metrics(y_test, weighted_score, threshold)
    row.update({
        "fold": idx,
        "model": "committee_weighted_soft_voting",
        "train_time_seconds": 0.0,
        "prediction_time_seconds": 0.0,
        "combination_time_seconds": float(weighted_combination_time),
        "end_to_end_time_seconds": float(committee_base_time + weighted_combination_time),
    })
    rows.append(row)

    meta = {
        "fold": idx,
        "train_positive_used": int(npos),
        "train_negative_used": int(nneg),
        "test_positive": int(len(test_pos)),
        "test_negative": int(len(test_neg)),
        "feature_batch_size": int(feature_batch_size),
        "weighted_voting_weights": dict(zip(names, w.tolist())),
        "train_times_seconds": {k: float(v) for k, v in train_times.items()},
        "prediction_times_seconds": {k: float(v) for k, v in prediction_times.items()},
        "committee_base_time_seconds": float(committee_base_time),
        "committee_combination_times_seconds": {
            "hard_voting": float(hard_combination_time),
            "soft_voting": float(soft_combination_time),
            "weighted_soft_voting": float(weighted_combination_time),
        },
    }

    del (
        models, embeddings, fold, probs, hard_votes, hard_pred, hard_score,
        soft_score, weighted_score, y_test
    )
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return pd.DataFrame(rows), meta


def evaluate_all_folds(
    folds_dir, embeddings_dir, output_dir="results",
    threshold=THRESHOLD, max_train_per_class=MAX_TRAIN_PER_CLASS,
    feature_batch_size=FEATURE_BATCH_SIZE, ensemble_weights=None,
):
    folds_dir, embeddings_dir, output_dir = map(Path, (folds_dir, embeddings_dir, output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    fold_paths = sorted(folds_dir.glob("fold_*.npz"), key=lambda p: int(p.stem.split("_")[-1]))
    if not fold_paths:
        raise FileNotFoundError(f"Nenhum fold_*.npz em {folds_dir}")

    all_rows, metadata = [], []
    for fp in fold_paths:
        idx = int(fp.stem.split("_")[-1])
        ep = embeddings_dir / f"fold_{idx}.pt"
        if not ep.exists():
            raise FileNotFoundError(ep)
        df, meta = evaluate_fold(
            fp, ep, threshold=threshold,
            max_train_per_class=max_train_per_class,
            feature_batch_size=feature_batch_size,
            ensemble_weights=ensemble_weights,
        )
        all_rows.append(df)
        metadata.append(meta)
        # Persist after every fold so an interruption does not lose completed work.
        pd.concat(all_rows, ignore_index=True).to_csv(output_dir / "metrics_per_fold.csv", index=False)
        with (output_dir / "run_metadata.json").open("w") as f:
            json.dump(metadata, f, indent=2)

    per_fold = pd.concat(all_rows, ignore_index=True)
    metric_cols = [
        "accuracy", "balanced_accuracy", "precision", "recall", "specificity",
        "f1", "f0.5", "f2", "roc_auc", "average_precision", "mcc", "log_loss",
    ]
    time_cols = [
        "train_time_seconds", "prediction_time_seconds",
        "combination_time_seconds", "end_to_end_time_seconds",
    ]
    summary = per_fold.groupby("model")[metric_cols + time_cols].agg(["mean", "std"])
    summary.to_csv(output_dir / "metrics_summary.csv")
    confusion = per_fold.groupby("model")[["tn", "fp", "fn", "tp"]].sum().reset_index()
    confusion.to_csv(output_dir / "confusion_matrices_aggregated.csv", index=False)

    print("\n=== MÉDIAS ± DESVIO-PADRÃO ===")
    for model in sorted(per_fold.model.unique()):
        s = per_fold[per_fold.model == model]
        print(f"\n[{model}]")
        for m in metric_cols:
            print(f"{m:>24}: {s[m].mean():.4f} ± {s[m].std(ddof=1):.4f}")
        print("  -- tempos (segundos) --")
        for t in time_cols:
            print(f"{t:>24}: {s[t].mean():.2f} ± {s[t].std(ddof=1):.2f}")
    return per_fold, summary, confusion

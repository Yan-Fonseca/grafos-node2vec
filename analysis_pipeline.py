import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier


SEED = 42
THRESHOLD = 0.5


def load_fold(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_embeddings(path):
    return torch.load(path, map_location="cpu", weights_only=True).float().numpy()


def edge_operator_hadamard(embeddings, edges):
    edges = np.asarray(edges, dtype=np.int64)
    if edges.size == 0:
        return np.empty((0, embeddings.shape[1]), dtype=np.float32)
    u = edges[:, 0]
    v = edges[:, 1]

    max_id = max(int(u.max()), int(v.max()))
    if max_id >= embeddings.shape[0]:
        raise IndexError(
            f"Aresta referencia nó {max_id}, mas embeddings possuem "
            f"apenas {embeddings.shape[0]} linhas."
        )
    return embeddings[u] * embeddings[v]


def build_xy(fold, embeddings):
    x_train_pos = edge_operator_hadamard(embeddings, fold["train_pos"])
    x_train_neg = edge_operator_hadamard(embeddings, fold["train_neg"])
    x_test_pos = edge_operator_hadamard(embeddings, fold["test_pos"])
    x_test_neg = edge_operator_hadamard(embeddings, fold["test_neg"])

    x_train = np.vstack([x_train_pos, x_train_neg])
    y_train = np.concatenate([
        np.ones(len(x_train_pos), dtype=np.int8),
        np.zeros(len(x_train_neg), dtype=np.int8),
    ])
    x_test = np.vstack([x_test_pos, x_test_neg])
    y_test = np.concatenate([
        np.ones(len(x_test_pos), dtype=np.int8),
        np.zeros(len(x_test_neg), dtype=np.int8),
    ])

    return x_train, y_train, x_test, y_test


def get_models(seed=SEED):
    return {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=2000,
                random_state=seed,
                n_jobs=None,
            )),
        ]),
        "random_forest": RandomForestClassifier(
            n_estimators=400,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        ),
        "xgboost": XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=seed,
            n_jobs=-1,
            tree_method="hist",
        ),
    }


def compute_metrics(y_true, probabilities, threshold=THRESHOLD):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predictions = (probabilities >= threshold).astype(np.int8)

    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()

    return {
        "accuracy": accuracy_score(y_true, predictions),
        "balanced_accuracy": balanced_accuracy_score(y_true, predictions),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        "f1": f1_score(y_true, predictions, zero_division=0),
        "f0.5": fbeta_score(y_true, predictions, beta=0.5, zero_division=0),
        "f2": fbeta_score(y_true, predictions, beta=2.0, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "average_precision": average_precision_score(y_true, probabilities),
        "mcc": matthews_corrcoef(y_true, predictions),
        "log_loss": log_loss(y_true, probabilities, labels=[0, 1]),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def evaluate_fold(
    fold_path,
    embedding_path,
    threshold=THRESHOLD,
    ensemble_weights=None,
    seed=SEED,
):
    fold = load_fold(fold_path)
    embeddings = load_embeddings(embedding_path)
    fold_idx = int(fold["fold"])

    x_train, y_train, x_test, y_test = build_xy(fold, embeddings)
    models = get_models(seed + fold_idx)

    probabilities = {}
    results = []

    for name, model in models.items():
        print(f"Fold {fold_idx} | treinando {name}...")
        model.fit(x_train, y_train)
        proba = model.predict_proba(x_test)[:, 1]
        probabilities[name] = proba

        metrics = compute_metrics(y_test, proba, threshold)
        metrics.update({"fold": fold_idx, "model": name})
        results.append(metrics)

    # Comitê por soft voting. Pesos iguais por padrão para não usar o conjunto
    # de teste na escolha dos pesos.
    model_names = list(probabilities.keys())
    if ensemble_weights is None:
        weights = np.ones(len(model_names), dtype=np.float64)
    else:
        weights = np.asarray([ensemble_weights[n] for n in model_names], dtype=np.float64)
        if np.any(weights < 0) or weights.sum() == 0:
            raise ValueError("Os pesos do comitê devem ser não-negativos e somar > 0.")

    weights = weights / weights.sum()
    committee_proba = np.average(
        np.column_stack([probabilities[n] for n in model_names]),
        axis=1,
        weights=weights,
    )

    committee_metrics = compute_metrics(y_test, committee_proba, threshold)
    committee_metrics.update({"fold": fold_idx, "model": "committee_soft_voting"})
    results.append(committee_metrics)

    return pd.DataFrame(results), {
        "fold": fold_idx,
        "model_names": model_names,
        "ensemble_weights": dict(zip(model_names, weights.tolist())),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
    }


def evaluate_all_folds(
    folds_dir,
    embeddings_dir,
    output_dir="results",
    threshold=THRESHOLD,
    ensemble_weights=None,
):
    folds_dir = Path(folds_dir)
    embeddings_dir = Path(embeddings_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fold_paths = sorted(
        folds_dir.glob("fold_*.pkl"),
        key=lambda p: int(p.stem.split("_")[-1]),
    )
    if not fold_paths:
        raise FileNotFoundError(f"Nenhum fold_*.pkl encontrado em {folds_dir}")

    all_results = []
    metadata = []

    for fold_path in fold_paths:
        fold_idx = int(fold_path.stem.split("_")[-1])
        embedding_path = embeddings_dir / f"fold_{fold_idx}.pt"
        if not embedding_path.exists():
            raise FileNotFoundError(
                f"Embedding do fold {fold_idx} não encontrado: {embedding_path}"
            )

        fold_df, fold_meta = evaluate_fold(
            fold_path=fold_path,
            embedding_path=embedding_path,
            threshold=threshold,
            ensemble_weights=ensemble_weights,
        )
        all_results.append(fold_df)
        metadata.append(fold_meta)

    per_fold = pd.concat(all_results, ignore_index=True)
    per_fold.to_csv(output_dir / "metrics_per_fold.csv", index=False)

    metric_cols = [
        "accuracy", "balanced_accuracy", "precision", "recall", "specificity",
        "f1", "f0.5", "f2", "roc_auc", "average_precision", "mcc", "log_loss",
    ]

    summary = (
        per_fold.groupby("model")[metric_cols]
        .agg(["mean", "std"])
        .sort_index()
    )
    summary.to_csv(output_dir / "metrics_summary.csv")

    # Matriz de confusão agregada entre os folds por modelo.
    confusion = (
        per_fold.groupby("model")[["tn", "fp", "fn", "tp"]]
        .sum()
        .reset_index()
    )
    confusion.to_csv(output_dir / "confusion_matrices_aggregated.csv", index=False)

    with (output_dir / "run_metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)

    print("\n=== MÉDIAS ± DESVIO-PADRÃO ENTRE FOLDS ===")
    for model in sorted(per_fold["model"].unique()):
        print(f"\n[{model}]")
        subset = per_fold[per_fold["model"] == model]
        for metric in metric_cols:
            print(
                f"{metric:>20}: "
                f"{subset[metric].mean():.4f} ± {subset[metric].std(ddof=1):.4f}"
            )

    return per_fold, summary, confusion


if __name__ == "__main__":
    FOLDS_DIR = "/home/souzajbr/grafos/folds_cache-astro-ph"
    EMBEDDINGS_DIR = "/home/souzajbr/grafos/embeddings/astro-ph"
    OUTPUT_DIR = "/home/souzajbr/grafos/results/astro-ph"

    evaluate_all_folds(
        folds_dir=FOLDS_DIR,
        embeddings_dir=EMBEDDINGS_DIR,
        output_dir=OUTPUT_DIR,
        threshold=0.5,
    )

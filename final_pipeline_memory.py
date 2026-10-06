"""End-to-end memory-efficient link prediction pipeline for large coauthorship graphs."""

from pathlib import Path

from folds_separation_memory import kfold_link_prediction_split_memory_efficient
from embedding_pipeline_memory import generate_all_embeddings
from analysis_pipeline_memory import evaluate_all_folds

DATASET = "coauth-DBLP"
BASE_DIR = Path("/home/souzajbr/grafos")
EDGE_LIST_PATH = BASE_DIR / "dataset" / f"{DATASET}.txt"
FOLDS_DIR = BASE_DIR / f"folds_compact-{DATASET}"
EMBEDDINGS_DIR = BASE_DIR / "embeddings" / DATASET
TEMP_DIR = BASE_DIR / "tmp_embedding_edges" / DATASET
RESULTS_DIR = BASE_DIR / "results" / DATASET

N_SPLITS = 5
NEG_RATIO = 1.0
SEED = 42
PRESERVE_CONNECTIVITY = True

# Scalability controls. All three classifiers receive the SAME sampled edges.
# 500k positives + 500k negatives at 128 float32 dimensions ~= 0.48 GiB X_train.
MAX_TRAIN_PER_CLASS = 500_000
FEATURE_BATCH_SIZE = 200_000


def main():
    print("\n[1/3] Gerando/carregando folds compactos...")
    kfold_link_prediction_split_memory_efficient(
        edge_list_path=str(EDGE_LIST_PATH),
        n_splits=N_SPLITS,
        neg_ratio=NEG_RATIO,
        random_state=SEED,
        preserve_connectivity=PRESERVE_CONNECTIVITY,
        save_dir=str(FOLDS_DIR),
        resume=True,
    )

    print("\n[2/3] Gerando embeddings independentes por fold...")
    generate_all_embeddings(
        folds_dir=str(FOLDS_DIR),
        embeddings_dir=str(EMBEDDINGS_DIR),
        temp_dir=str(TEMP_DIR),
        overwrite=False,
    )

    print("\n[3/3] Treinando classificadores e comitê...")
    evaluate_all_folds(
        folds_dir=str(FOLDS_DIR),
        embeddings_dir=str(EMBEDDINGS_DIR),
        output_dir=str(RESULTS_DIR),
        threshold=0.5,
        max_train_per_class=MAX_TRAIN_PER_CLASS,
        feature_batch_size=FEATURE_BATCH_SIZE,
    )
    print(f"\nResultados: {RESULTS_DIR}")


if __name__ == "__main__":
    main()

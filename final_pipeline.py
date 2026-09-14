"""Pipeline completo: folds -> embeddings por fold -> classificadores -> comitê."""

from pathlib import Path

from folds_separation import kfold_link_prediction_split
from embedding_pipeline import generate_all_embeddings
from analysis_pipeline import evaluate_all_folds


DATASET = "ca-AstroPh"
EDGE_LIST_PATH = f"/home/souzajbr/grafos/dataset/{DATASET}.txt"
BASE_DIR = Path("/home/souzajbr/grafos")

FOLDS_DIR = BASE_DIR / f"folds_cache-{DATASET}"
EMBEDDINGS_DIR = BASE_DIR / "embeddings" / DATASET
RESULTS_DIR = BASE_DIR / "results" / DATASET

N_SPLITS = 5
NEG_RATIO = 1.0
SEED = 42
PRESERVE_CONNECTIVITY = True


def main():
    print("\n[1/3] Gerando/carregando folds...")
    kfold_link_prediction_split(
        edge_list_path=EDGE_LIST_PATH,
        n_splits=N_SPLITS,
        neg_ratio=NEG_RATIO,
        random_state=SEED,
        preserve_connectivity=PRESERVE_CONNECTIVITY,
        save_dir=str(FOLDS_DIR),
        resume=True,
    )

    print("\n[2/3] Gerando embeddings independentes para cada fold...")
    generate_all_embeddings(
        folds_dir=str(FOLDS_DIR),
        embeddings_dir=str(EMBEDDINGS_DIR),
        overwrite=False,
    )

    print("\n[3/3] Treinando classificadores e comitê...")
    evaluate_all_folds(
        folds_dir=str(FOLDS_DIR),
        embeddings_dir=str(EMBEDDINGS_DIR),
        output_dir=str(RESULTS_DIR),
        threshold=0.5,
    )

    print(f"\nResultados salvos em: {RESULTS_DIR}")


if __name__ == "__main__":
    main()

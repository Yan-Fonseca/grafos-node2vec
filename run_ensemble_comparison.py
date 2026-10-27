"""Executa somente a etapa de classificação/ensembles.

Reaproveita folds .npz e embeddings .pt já existentes, sem regenerar
folds ou embeddings.
"""

from pathlib import Path

from analysis_pipeline_ensembles_timing import evaluate_all_folds

DATASET = "coauth-DBLP"
BASE_DIR = Path("/home/souzajbr/grafos")

FOLDS_DIR = BASE_DIR / f"folds_compact-{DATASET}"
EMBEDDINGS_DIR = BASE_DIR / "embeddings" / DATASET
RESULTS_DIR = BASE_DIR / "results" / f"{DATASET}_ensemble_comparison_timing"

THRESHOLD = 0.5
MAX_TRAIN_PER_CLASS = 500_000
FEATURE_BATCH_SIZE = 200_000

# IMPORTANTE:
# Estes pesos são apenas uma configuração experimental inicial.
# Para resultados científicos finais, defina-os com base SOMENTE em
# treino/validação, nunca usando o conjunto de teste do fold avaliado.
# A soma não precisa ser 1: o código normaliza automaticamente.
WEIGHTED_VOTING_WEIGHTS = {
    "logistic_regression": 0.20,
    "random_forest": 0.30,
    "xgboost": 0.50,
}


def main():
    print("=" * 72)
    print("AVALIAÇÃO DOS CLASSIFICADORES E COMITÊS")
    print("=" * 72)
    print(f"Folds      : {FOLDS_DIR}")
    print(f"Embeddings : {EMBEDDINGS_DIR}")
    print(f"Resultados : {RESULTS_DIR}")
    print(f"Threshold  : {THRESHOLD}")
    print(f"Pesos      : {WEIGHTED_VOTING_WEIGHTS}")
    print()

    evaluate_all_folds(
        folds_dir=str(FOLDS_DIR),
        embeddings_dir=str(EMBEDDINGS_DIR),
        output_dir=str(RESULTS_DIR),
        threshold=THRESHOLD,
        max_train_per_class=MAX_TRAIN_PER_CLASS,
        feature_batch_size=FEATURE_BATCH_SIZE,
        ensemble_weights=WEIGHTED_VOTING_WEIGHTS,
    )

    print("\nConcluído.")
    print(f"Resultados salvos em: {RESULTS_DIR}")


if __name__ == "__main__":
    main()

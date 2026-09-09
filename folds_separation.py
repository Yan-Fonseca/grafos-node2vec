"""
K-Fold split para Link Prediction em grafos de colaboração científica.

Dataset esperado: um arquivo de texto puro (sem cabeçalho) contendo APENAS
as arestas do grafo, uma por linha, com os dois nós separados por espaço
ou tab, por exemplo:
    1 2
    1 3
    2 3
    3 4

Estratégia:
    - As arestas POSITIVAS existentes são divididas em k folds via KFold.
    - Para cada fold, o grafo de treino é construído com todos os nós do
      grafo original, mas apenas com as arestas de treino (as arestas de
      teste são "removidas" para simular predição).
    - Arestas NEGATIVAS (pares de nós sem conexão) são amostradas
      aleatoriamente para treino e teste, evitando reaproveitar o mesmo
      par negativo em folds/conjuntos diferentes.
"""

import random

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


def load_graph(edge_list_path, sep=r"\s+"):
    """
    Carrega um dataset de arestas em formato de texto puro, sem cabeçalho,
    no estilo:
        1 2
        1 3
        2 3
        3 4

    Cada linha vira uma aresta (nó_a, nó_b). `sep=r"\s+"` aceita espaço(s)
    ou tab entre as colunas.
    """
    df = pd.read_csv(edge_list_path, sep=sep, header=None,
                      names=["source", "target"], engine="python")
    G = nx.from_pandas_edgelist(df, source="source", target="target")
    return G


def negative_sampling(G, n_samples, exclude_pairs=None, seed=None):
    """
    Amostra pares de nós SEM conexão no grafo (arestas negativas).

    Parameters
    ----------
    G : networkx.Graph
        Grafo completo, usado para saber quais pares já são arestas reais.
    n_samples : int
        Quantidade de arestas negativas a gerar.
    exclude_pairs : set[frozenset], opcional
        Pares adicionais a evitar (ex.: negativos já usados em outro fold),
        para não reaproveitar o mesmo par negativo em conjuntos diferentes.
    seed : int, opcional
        Semente para reprodutibilidade.

    Returns
    -------
    list[tuple]
        Lista de pares (u, v) que não são arestas em G.
    """
    rng = random.Random(seed)
    nodes = list(G.nodes())
    forbidden = set(map(frozenset, G.edges()))
    if exclude_pairs:
        forbidden |= exclude_pairs

    negatives = set()
    max_attempts = n_samples * 50
    attempts = 0
    while len(negatives) < n_samples and attempts < max_attempts:
        u, v = rng.sample(nodes, 2)
        pair = frozenset((u, v))
        if pair not in forbidden and pair not in negatives:
            negatives.add(pair)
        attempts += 1

    if len(negatives) < n_samples:
        print(
            f"[Aviso] Só foi possível gerar {len(negatives)} de "
            f"{n_samples} negativos solicitados (grafo pode estar denso demais)."
        )

    return [tuple(pair) for pair in negatives]


def kfold_link_prediction_split(
    edge_list_path,
    n_splits=5,
    neg_ratio=1.0,
    random_state=42,
):
    """
    Gera k folds para a tarefa de Link Prediction.

    Cada fold é um dicionário contendo:
        - 'G_train'  : grafo NetworkX de treino (todos os nós, arestas de treino)
        - 'train_pos': arestas positivas de treino
        - 'train_neg': arestas negativas amostradas para treino
        - 'test_pos' : arestas positivas de teste
        - 'test_neg' : arestas negativas amostradas para teste

    Parameters
    ----------
    neg_ratio : float
        Proporção negativos/positivos (1.0 = mesma quantidade de cada).
    """
    G_full = load_graph(edge_list_path)
    all_nodes = list(G_full.nodes())
    all_edges = np.array(list(G_full.edges()), dtype=object)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    folds = []
    used_negatives = set()  # evita reaproveitar o mesmo par negativo entre folds

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(all_edges)):
        train_pos = all_edges[train_idx]
        test_pos = all_edges[test_idx]

        # Grafo de treino: mesmos nós do grafo completo, só as arestas de treino
        G_train = nx.Graph()
        G_train.add_nodes_from(all_nodes)
        G_train.add_edges_from(train_pos)

        n_train_neg = int(len(train_pos) * neg_ratio)
        n_test_neg = int(len(test_pos) * neg_ratio)

        train_neg = negative_sampling(
            G_full, n_train_neg,
            exclude_pairs=used_negatives,
            seed=random_state + fold_idx,
        )
        used_negatives.update(map(frozenset, train_neg))

        test_neg = negative_sampling(
            G_full, n_test_neg,
            exclude_pairs=used_negatives,
            seed=random_state + fold_idx + 1000,
        )
        used_negatives.update(map(frozenset, test_neg))

        folds.append(
            {
                "fold": fold_idx,
                "G_train": G_train,
                "train_pos": train_pos.tolist(),
                "train_neg": train_neg,
                "test_pos": test_pos.tolist(),
                "test_neg": test_neg,
            }
        )

        print(
            f"Fold {fold_idx}: "
            f"train_pos={len(train_pos)}, train_neg={len(train_neg)}, "
            f"test_pos={len(test_pos)}, test_neg={len(test_neg)}"
        )

    return folds


if __name__ == "__main__":
    # ---- Ajuste estes parâmetros para o seu dataset ----
    EDGE_LIST_PATH = "colaboracao.txt"   # caminho do seu arquivo de arestas
    N_SPLITS = 5                         # número de folds
    NEG_RATIO = 1.0                      # 1 negativo para cada positivo

    folds = kfold_link_prediction_split(
        edge_list_path=EDGE_LIST_PATH,
        n_splits=N_SPLITS,
        neg_ratio=NEG_RATIO,
        random_state=42,
    )

    # Exemplo de acesso aos dados do fold 0
    fold0 = folds[0]
    print("\nExemplo - Fold 0:")
    print("Nº de nós no grafo de treino:", fold0["G_train"].number_of_nodes())
    print("Nº de arestas de treino (pos):", len(fold0["train_pos"]))
    print("Nº de arestas de teste (pos):", len(fold0["test_pos"]))

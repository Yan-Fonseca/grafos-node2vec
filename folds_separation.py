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

import os
import pickle
import random

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


def load_graph(edge_list_path, sep=r"\s+"):
    r"""
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


def check_isolated_test_nodes(G_train, test_pos):
    """
    Verifica quantos nós envolvidos nas arestas de TESTE ficaram isolados
    (grau 0) no grafo de TREINO — ou seja, perderam todas as suas conexões
    ao remover as arestas de teste.

    Isso é importante para métodos como node2vec/DeepWalk: um nó isolado
    em G_train não participa de nenhum random walk, então seu embedding
    fica essencialmente não-treinado (aleatório), o que compromete a
    predição de qualquer aresta envolvendo esse nó.

    Parameters
    ----------
    G_train : networkx.Graph
        Grafo de treino do fold (já sem as arestas de teste).
    test_pos : array-like
        Lista/array de arestas positivas de teste, formato [(u, v), ...].

    Returns
    -------
    dict
        {
            'test_nodes': int,          # nós distintos envolvidos no teste
            'isolated_nodes': int,      # quantos desses ficaram com grau 0
            'isolated_pct': float,      # percentual (0-100)
            'isolated_node_list': list, # os próprios nós isolados
        }
    """
    test_nodes = set()
    for u, v in test_pos:
        test_nodes.add(u)
        test_nodes.add(v)

    isolated = [
        node for node in test_nodes
        if node in G_train and G_train.degree(node) == 0
    ]

    n_test_nodes = len(test_nodes)
    n_isolated = len(isolated)
    pct = (n_isolated / n_test_nodes * 100) if n_test_nodes > 0 else 0.0

    return {
        "test_nodes": n_test_nodes,
        "isolated_nodes": n_isolated,
        "isolated_pct": pct,
        "isolated_node_list": isolated,
    }


def check_connectivity(G_train):
    """
    Relata o estado de conectividade de G_train: quantos componentes
    conexos existem e qual fração dos nós está no maior componente.

    Um número alto de componentes (ou um maior componente pequeno em
    relação ao total de nós) indica que o fold fragmentou bastante o
    grafo original — o que prejudica especialmente node2vec/DeepWalk e
    GNNs baseadas em message passing (GCN, GraphSAGE, GAT, GAE/VGAE),
    já que essas técnicas não propagam informação entre componentes
    distintos.

    Returns
    -------
    dict
        {
            'n_components': int,
            'largest_component_size': int,
            'largest_component_pct': float,  # % dos nós no maior componente
        }
    """
    components = list(nx.connected_components(G_train))
    n_components = len(components)
    largest = max((len(c) for c in components), default=0)
    total_nodes = G_train.number_of_nodes()
    pct = (largest / total_nodes * 100) if total_nodes else 0.0

    return {
        "n_components": n_components,
        "largest_component_size": largest,
        "largest_component_pct": pct,
    }


def get_spanning_forest_edges(G):
    """
    Calcula uma floresta geradora de G (uma árvore geradora por componente
    conexo) e retorna o conjunto de arestas que a compõem, como frozensets.

    Se essas arestas forem sempre mantidas no treino (nunca sorteadas para
    o teste), G_train preserva a MESMA estrutura de componentes conexos
    que o grafo original G — ou seja, nenhum fold fragmenta o grafo além
    do que ele já era originalmente. É a técnica padrão na literatura de
    link prediction para evitar desconexão artificial induzida pelo split.
    """
    protected = set()
    for component in nx.connected_components(G):
        sub = G.subgraph(component)
        T = nx.minimum_spanning_tree(sub)
        protected.update(frozenset(e) for e in T.edges())
    return protected


def kfold_link_prediction_split(
    edge_list_path,
    n_splits=5,
    neg_ratio=1.0,
    random_state=42,
    preserve_connectivity=False,
    save_dir=None,
    resume=True,
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
    preserve_connectivity : bool
        Se True, calcula uma floresta geradora do grafo completo e nunca
        sorteia essas arestas para o conjunto de teste, garantindo que
        G_train nunca fique mais fragmentado do que o grafo original.
        Recomendado para node2vec/DeepWalk e GNNs baseadas em message
        passing (GCN, GraphSAGE, GAT, GAE/VGAE). Reduz um pouco o pool de
        arestas elegíveis para teste (as arestas da floresta geradora
        ficam sempre no treino).
    save_dir : str, opcional
        Se fornecido, salva cada fold em '{save_dir}/fold_{idx}.pkl' assim
        que ele é criado (não espera todos os folds terminarem). Útil para
        não perder progresso se o processo for interrompido.
    resume : bool
        Se True (padrão) e save_dir for fornecido, pula o reprocessamento
        de qualquer fold cujo arquivo já exista em save_dir, carregando-o
        do disco em vez de recalculá-lo do zero.
    """
    G_full = load_graph(edge_list_path)
    all_nodes = list(G_full.nodes())
    all_edges_list = list(G_full.edges())

    if preserve_connectivity:
        protected_set = get_spanning_forest_edges(G_full)
        protected_edges = [e for e in all_edges_list if frozenset(e) in protected_set]
        candidate_edges = np.array(
            [e for e in all_edges_list if frozenset(e) not in protected_set],
            dtype=object,
        )
        print(
            f"[preserve_connectivity=True] {len(protected_edges)} arestas "
            f"protegidas (floresta geradora) ficarão sempre no treino; "
            f"{len(candidate_edges)} arestas elegíveis para teste."
        )
    else:
        protected_edges = []
        candidate_edges = np.array(all_edges_list, dtype=object)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    folds = []
    used_negatives = set()  # evita reaproveitar o mesmo par negativo entre folds

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(candidate_edges)):

        # --- Retomada: se este fold já foi salvo, carrega em vez de recalcular ---
        if save_dir is not None and resume:
            fold_path = os.path.join(save_dir, f"fold_{fold_idx}.pkl")
            if os.path.exists(fold_path):
                fold = load_single_fold(fold_idx, save_dir)
                folds.append(fold)
                # mantém o controle de negativos já usados consistente
                used_negatives.update(map(frozenset, fold["train_neg"]))
                used_negatives.update(map(frozenset, fold["test_neg"]))
                print(f"Fold {fold_idx}: já existe em '{fold_path}', pulando recomputação.")
                continue

        train_pos = np.array(
            protected_edges + candidate_edges[train_idx].tolist(), dtype=object
        )
        test_pos = candidate_edges[test_idx]

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

        isolation_report = check_isolated_test_nodes(G_train, test_pos)
        connectivity_report = check_connectivity(G_train)

        folds.append(
            {
                "fold": fold_idx,
                "G_train": G_train,
                "train_pos": train_pos.tolist(),
                "train_neg": train_neg,
                "test_pos": test_pos.tolist(),
                "test_neg": test_neg,
                "isolation_report": isolation_report,
                "connectivity_report": connectivity_report,
            }
        )

        if save_dir is not None:
            save_single_fold(folds[-1], save_dir)

        print(
            f"Fold {fold_idx}: "
            f"train_pos={len(train_pos)}, train_neg={len(train_neg)}, "
            f"test_pos={len(test_pos)}, test_neg={len(test_neg)}"
        )
        print(
            f"          nós isolados em G_train (entre os de teste): "
            f"{isolation_report['isolated_nodes']}/{isolation_report['test_nodes']} "
            f"({isolation_report['isolated_pct']:.1f}%)"
        )
        print(
            f"          componentes conexos em G_train: "
            f"{connectivity_report['n_components']} "
            f"(maior componente: {connectivity_report['largest_component_pct']:.1f}% dos nós)"
        )
        if isolation_report["isolated_nodes"] > 0:
            print(
                "          [Aviso] embeddings transdutivos (node2vec/DeepWalk) "
                "terão qualidade comprometida para esses nós."
            )
        if connectivity_report["n_components"] > 1:
            print(
                "          [Aviso] grafo fragmentado — node2vec/DeepWalk e "
                "GNNs (GCN/GraphSAGE/GAT/GAE) não propagam informação entre "
                "componentes distintos."
            )

    avg_isolated_pct = np.mean(
        [f["isolation_report"]["isolated_pct"] for f in folds]
    )
    print(
        f"\nMédia de nós de teste isolados em G_train, entre todos os folds: "
        f"{avg_isolated_pct:.1f}%"
    )
    if avg_isolated_pct > 5:
        print(
            "[Aviso] Percentual relevante de nós isolados — considere usar "
            "heurísticas topológicas (Adamic-Adar, Common Neighbors) como "
            "baseline ou principal método, já que dependem menos de "
            "conectividade prévia do nó."
        )

    return folds


def save_single_fold(fold, save_dir="folds_cache"):
    """
    Salva UM fold individualmente em '{save_dir}/fold_{idx}.pkl'.

    Chamada dentro do loop de geração, logo após o fold ser criado —
    assim, se o processo for interrompido no meio (ex: fold 7 de 10),
    os folds já processados não se perdem.
    """
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, f"fold_{fold['fold']}.pkl")
    with open(filepath, "wb") as f:
        pickle.dump(fold, f)
    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"          -> fold salvo em '{filepath}' ({size_mb:.2f} MB)")


def load_single_fold(fold_idx, save_dir="folds_cache"):
    """Carrega um único fold salvo por save_single_fold()."""
    filepath = os.path.join(save_dir, f"fold_{fold_idx}.pkl")
    with open(filepath, "rb") as f:
        fold = pickle.load(f)
    return fold


def load_all_folds(save_dir="folds_cache"):
    """
    Carrega todos os folds de um diretório salvo por save_single_fold(),
    em ordem (fold_0.pkl, fold_1.pkl, ...).
    """
    fold_files = sorted(
        f for f in os.listdir(save_dir)
        if f.startswith("fold_") and f.endswith(".pkl")
    )
    folds = []
    for fname in fold_files:
        with open(os.path.join(save_dir, fname), "rb") as f:
            folds.append(pickle.load(f))
    print(f"{len(folds)} folds carregados de '{save_dir}'.")
    return folds


def save_folds(folds, filepath="folds_cache.pkl"):
    """
    Salva a lista de folds (com G_train, arestas positivas/negativas e
    relatórios de diagnóstico) em disco via pickle, evitando ter que
    reprocessar tudo (negative sampling + spanning forest + diagnósticos)
    novamente em execuções futuras.

    Atenção: o arquivo pode ficar grande em grafos maiores, já que cada
    fold guarda uma cópia completa do grafo G_train. Se isso for um
    problema de espaço, considere salvar apenas as listas de arestas
    (train_pos/train_neg/test_pos/test_neg) e reconstruir G_train ao
    carregar, em vez do grafo já pronto.
    """
    with open(filepath, "wb") as f:
        pickle.dump(folds, f)
    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"Folds salvos em '{filepath}' ({size_mb:.1f} MB).")


def load_folds(filepath="folds_cache.pkl"):
    """Carrega uma lista de folds previamente salva com save_folds()."""
    with open(filepath, "rb") as f:
        folds = pickle.load(f)
    print(f"Folds carregados de '{filepath}' ({len(folds)} folds).")
    return folds


if __name__ == "__main__":
    # ---- Ajuste estes parâmetros para o seu dataset ----
    EDGE_LIST_PATH = "/home/souzajbr/grafos/dataset/ca-AstroPh.txt"   # caminho do seu arquivo de arestas
    N_SPLITS = 5                         # número de folds
    NEG_RATIO = 1.0                      # 1 negativo para cada positivo
    PRESERVE_CONNECTIVITY = True         # evita fragmentar o grafo nos folds
    SAVE_DIR = "/home/souzajbr/grafos/folds_cache-astro-ph"             # cada fold vai para folds_cache/fold_N.pkl

    folds = kfold_link_prediction_split(
        edge_list_path=EDGE_LIST_PATH,
        n_splits=N_SPLITS,
        neg_ratio=NEG_RATIO,
        random_state=42,
        preserve_connectivity=PRESERVE_CONNECTIVITY,
        save_dir=SAVE_DIR,   # salva (e retoma) fold a fold automaticamente
        resume=True,
    )

    if os.path.exists(FOLDS_CACHE_PATH):
        # Já existe um cache — carrega em vez de reprocessar tudo de novo
        print('Carregando arquivo de folds salvo')
        folds = load_folds(FOLDS_CACHE_PATH)
    else:
        folds = kfold_link_prediction_split(
            edge_list_path=EDGE_LIST_PATH,
            n_splits=N_SPLITS,
            neg_ratio=NEG_RATIO,
            random_state=42,
            preserve_connectivity=PRESERVE_CONNECTIVITY,
        )
        save_folds(folds, FOLDS_CACHE_PATH)

    # Se quiser recarregar depois, em outra execução, sem rodar tudo de novo:
    # folds = load_all_folds(SAVE_DIR)

    # Exemplo de acesso aos dados do fold 0
    fold0 = folds[0]
    print("\nExemplo - Fold 0:")
    print("Nº de nós no grafo de treino:", fold0["G_train"].number_of_nodes())
    print("Nº de arestas de treino (pos):", len(fold0["train_pos"]))
    print("Nº de arestas de teste (pos):", len(fold0["test_pos"]))

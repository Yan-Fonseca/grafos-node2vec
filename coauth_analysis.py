import os 
import time 
from collections import defaultdict 
import igraph as ig 
import pandas as pd 

# ============================================================ 
# CONFIGURAÇÕES 
# ============================================================ 

INPUT_FILE = "/home/souzajbr/grafos/dataset/coauth-DBLP.txt" 
OUTPUT_DIR = "/home/souzajbr/grafos/dataset/metricas_rede" 

# Número de vértices utilizados como amostra para as métricas 
# aproximadas de betweenness e closeness. 
# 
# None = cálculo exato 
# 1000 = aproximadamente 1000 vértices 
# 
# Para redes grandes, recomendo começar com 1000. 

BETWEENNESS_SAMPLES = 1000 
CLOSENESS_SAMPLES = 1000 

# Quantidade de pesquisadores que serão mostrados 
# nos rankings. TOP_N = 100

# ============================================================ 
# PREPARAÇÃO 
# ============================================================ 

os.makedirs(OUTPUT_DIR, exist_ok=True) 

print("=" * 60) 
print("ANÁLISE DE REDE DE COAUTORIA") 
print("=" * 60)

# ============================================================ 
# 1. LEITURA E AGREGAÇÃO DAS ARESTAS 
# ============================================================ 

print("\n[1/10] Lendo arquivo...") 
start = time.time() 
edge_weights = defaultdict(int) 

with open(INPUT_FILE, "r", encoding="ascii") as f: 
    for line_number, line in enumerate(f, start=1): 
        line = line.strip() 
        if not line: 
            continue 
        
        parts = line.split() 
        if len(parts) < 2: 
            continue 
        try: 
            u = int(parts[0]) 
            v = int(parts[1]) 
        except ValueError: 
            continue 
        
        # Não permitir self-loop 
        if u == v: 
            continue 
        
        # Rede não direcionada: 
        # 1 2 == 2 1 
        if u > v: 
            u, v = v, u 
            edge_weights[(u, v)] += 1 

print(f"Tempo de leitura: {time.time() - start:.2f}s") 
print(f"Arestas únicas: {len(edge_weights):,}")

# ============================================================ 
# 2. CRIAÇÃO DO GRAFO 
# ============================================================ 

print("\n[2/10] Construindo grafo...") 
start = time.time() 

# Lista de arestas 
edges = list(edge_weights.keys()) 

# Nós existentes 
vertices = sorted( set( node for edge in edges for node in edge ) ) 

# Mapeamento: 
# identificador original -> índice interno 
node_to_index = { node: i for i, node in enumerate(vertices) } 

# Converter os identificadores para índices 
igraph_edges = [ ( node_to_index[u], node_to_index[v] ) for u, v in edges ] 

# Criar grafo 
G = ig.Graph( n=len(vertices), edges=igraph_edges, directed=False ) 

# Identificador original do pesquisador 
G.vs["node"] = vertices 

# Peso = número de colaborações 
G.es["weight"] = [ edge_weights[edge] for edge in edges ] 

print(f"Tempo: {time.time() - start:.2f}s") 
print(f"Nós: {G.vcount():,}") 
print(f"Arestas únicas: {G.ecount():,}")

# ============================================================ 
# 3. ESTATÍSTICAS BÁSICAS 
# ============================================================ 
print("\n[3/10] Calculando estatísticas básicas...") 
n_nodes = G.vcount() 
n_edges = G.ecount() 

if n_nodes > 1: 
    density = ( 2 * n_edges / (n_nodes * (n_nodes - 1)) ) 
else: 
    density = 0

degree = G.degree() 
weighted_degree = G.strength( weights="weight" ) 
average_degree = ( sum(degree) / n_nodes if n_nodes > 0 else 0 )

# ============================================================ 
# 4. COMPONENTES CONECTADOS 
# ============================================================ 
print("\n[4/10] Calculando componentes conectados...") 

components = G.connected_components() 
component_sizes = sorted( components.sizes(), reverse=True ) 
n_components = len(component_sizes) 

largest_component = ( component_sizes[0] if component_sizes else 0 ) 
largest_component_fraction = ( largest_component / n_nodes if n_nodes > 0 else 0 ) 

print(f"Componentes: {n_components:,}") 
print(f"Maior componente: {largest_component:,} nós") 
print( f"Proporção na maior componente: " f"{largest_component_fraction:.4f}" )

# ============================================================ 
# 5. CLUSTERING COEFFICIENT 
# ============================================================ 
print("\n[5/10] Calculando clustering coefficient...") 
start = time.time() 
local_clustering = G.transitivity_local_undirected( vertices=None, mode="zero" ) 
global_clustering = G.transitivity_avglocal_undirected( mode="zero" ) 

print( f"Clustering médio: " f"{global_clustering:.6f}" ) 
print(f"Tempo: {time.time() - start:.2f}s")

# ============================================================ 
# 6. PAGERANK 
# ============================================================ 
print("\n[6/10] Calculando PageRank...") 

start = time.time() 
pagerank = G.pagerank( directed=False, weights="weight" ) 

print(f"Tempo: {time.time() - start:.2f}s")

# ============================================================ 
# 7. EIGENVECTOR CENTRALITY 
# ============================================================ 
print("\n[7/10] Calculando eigenvector centrality...") 

start = time.time() 
try: 
    eigenvector = G.eigenvector_centrality( directed=False, weights="weight", scale=True ) 
except Exception as e: 
    print("Erro no eigenvector centrality:") 
    print(e) 
    eigenvector = [float("nan")] * n_nodes 

print(f"Tempo: {time.time() - start:.2f}s")

# ============================================================ 
# 8. K-CORE 
# ============================================================ 
print("\n[8/10] Calculando k-core...") 

start = time.time() 
coreness = G.coreness() 
print(f"Tempo: {time.time() - start:.2f}s")

# ============================================================ 
# 9. BETWEENNESS 
# ============================================================ 
print("\n[9/10] Calculando betweenness...") 
start = time.time() 

if BETWEENNESS_SAMPLES is None: 
    betweenness = G.betweenness( directed=False ) 
else: 
    # Seleciona uma amostra de vértices 
    # uniformemente distribuída. 
    if BETWEENNESS_SAMPLES >= n_nodes: 
        sources = list(range(n_nodes)) 
    else: 
        step = n_nodes / BETWEENNESS_SAMPLES 
        sources = [ int(i * step) for i in range(BETWEENNESS_SAMPLES) ] 
    betweenness = G.betweenness( directed=False, sources=sources ) 

    # Normalização aproximada 
    if len(sources) < n_nodes and n_nodes > 2: 
        scale = n_nodes / len(sources) 
        betweenness = [ value * scale for value in betweenness ] 

print(f"Tempo: {time.time() - start:.2f}s")

# ============================================================ 
# 10. CLOSENESS 
# ============================================================ 
print("\n[10/10] Calculando closeness...") 

start = time.time() 

# Para redes desconectadas, calculamos closeness por 
# componente para evitar problemas com distâncias infinitas. 
closeness = [0.0] * n_nodes 

for component in components: 
    component_vertices = list(component) 
    size = len(component_vertices) 
    if size <= 1: 
        closeness[component_vertices[0]] = 0.0 
        continue 
    
    # Cálculo exato para componentes pequenas 
    if ( CLOSENESS_SAMPLES is None or size <= CLOSENESS_SAMPLES ): 
        values = G.closeness( vertices=component_vertices, normalized=True ) 
        for vertex, value in zip( component_vertices, values ): 
            closeness[vertex] = value 
    else: 
        # Aproximação para componentes grandes 
        step = size / CLOSENESS_SAMPLES 
        sources = [ component_vertices[ int(i * step) ] for i in range(CLOSENESS_SAMPLES) ] 
        
        # Distâncias a partir das amostras 
        distances = G.distances( source=sources, target=component_vertices ) 

        for local_index, vertex in enumerate( component_vertices ): 
            total_distance = 0 
            count = 0 
            for row in distances: 
                distance = row[local_index] 
                if distance != float("inf"): 
                    total_distance += distance 
                    count += 1 
            if total_distance > 0: 
                closeness[vertex] = ( count / total_distance ) 

print(f"Tempo: {time.time() - start:.2f}s")

# ============================================================ 
# 11. CRIAÇÃO DO DATAFRAME FINAL 
# ============================================================ 
print("\nCriando tabela de métricas...") 

results = pd.DataFrame({ 
    "node": vertices, 
    "degree": degree, 
    "weighted_degree": weighted_degree, 
    "degree_centrality": [ d / (n_nodes - 1) if n_nodes > 1 else 0 for d in degree ],
    "betweenness": betweenness, 
    "closeness": closeness, 
    "pagerank": pagerank, 
    "eigenvector": eigenvector, 
    "clustering_coefficient": local_clustering, 
    "coreness": coreness 
})

# ============================================================ 
# 12. SALVAR TODAS AS MÉTRICAS 
# ============================================================ 
output_file = os.path.join( OUTPUT_DIR, "node_metrics.csv" ) 

results.to_csv( output_file, index=False ) 

print(f"\nTabela completa salva em:") 
print(output_file)

# ============================================================ 
# 13. TOP N PESQUISADORES 
# ============================================================ 
metrics = [ "degree", 
        "weighted_degree", 
        "degree_centrality", 
        "betweenness", 
        "closeness", 
        "pagerank", 
        "eigenvector", 
        "clustering_coefficient", 
        "coreness" 
] 

for metric in metrics: 
    filename = os.path.join( OUTPUT_DIR, f"top_{metric}.csv" ) 

    ( results[ [ "node", metric ] ] 
            .sort_values( metric, ascending=False ) 
            .head(TOP_N) 
            .to_csv( filename, index=False ) 
    )


# ============================================================ 
# 14. RESUMO DA REDE 
# ============================================================ 
summary_file = os.path.join( OUTPUT_DIR, "network_summary.txt" ) 

with open( summary_file, "w", encoding="utf-8" ) as f: 
    f.write("REDE DE COAUTORIA\n") 
    f.write("=" * 50 + "\n\n") 
    f.write( f"Número de pesquisadores: " 
            f"{n_nodes:,}\n" ) 
    f.write( f"Número de arestas únicas: " 
            f"{n_edges:,}\n" ) 
    f.write( f"Densidade: " 
            f"{density:.10f}\n" ) 
    f.write( f"Grau médio: " 
            f"{average_degree:.6f}\n" ) 
    f.write( f"Número de componentes: " 
            f"{n_components:,}\n" ) 
    f.write( f"Maior componente: " 
            f"{largest_component:,}\n" ) 
    f.write( f"Proporção da maior componente: " 
            f"{largest_component_fraction:.6f}\n" ) 
    f.write( f"Clustering coefficient médio: " 
            f"{global_clustering:.6f}\n" ) 
    f.write( f"Grau máximo: " 
            f"{max(degree) if degree else 0}\n" ) 
    f.write( f"Grau ponderado máximo: " 
            f"{max(weighted_degree) if weighted_degree else 0}\n" )
    f.write( f"K-core máximo: " 
            f"{max(coreness) if coreness else 0}\n" )

# ============================================================ 
# 15. FINAL 
# ============================================================ 
print("\n" + "=" * 60) 
print("ANÁLISE CONCLUÍDA") 
print("=" * 60) 

print(f"\nResultados em:") 
print(f" {OUTPUT_DIR}/") 
print("\nArquivos principais:") 
print(" node_metrics.csv") 
print(" network_summary.txt") 
print("\nRankings:") 

for metric in metrics: 
    print(f" top_{metric}.csv")

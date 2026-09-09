import argparse


def normalize_edges(input_file, output_file):
    edges = set()

    with open(input_file, "r") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            u, v = line.split()[:2]

            # Normaliza a ordem dos nós.
            # Assim, "1 2" e "2 1" são consideradas a mesma aresta.
            edge = (int(u), int(v))
            edge = tuple(sorted(edge))

            edges.add(edge)

    # Ordena para que o arquivo de saída fique organizado
    edges = sorted(edges)
        
    with open(output_file, "w") as f:
        for u, v in edges:
            f.write(f"{u} {v}\n")

    print(f"Arestas originais: arquivo processado")
    print(f"Arestas únicas: {len(edges)}")
    print(f"Arquivo salvo em: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Remove arestas duplicadas de um arquivo de grafo."
    )

    parser.add_argument(
        "input",
        help="Arquivo de entrada"
    )

    parser.add_argument(
        "output",
        help="Arquivo de saída"
    )

    args = parser.parse_args()

    normalize_edges(args.input, args.output)

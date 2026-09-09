import json
from pathlib import Path
from itertools import combinations

PATH = Path('~/grafos/dataset/coauth-DBLP.json').expanduser()

class Pipe:
    def __init__(self):
        self.dados = None

    def import_json(self, path):
        if not path.is_file(): 
            raise FileNotFoundError(f"Arquivo não encontrado: {path}")
        try:
            with open(path, 'r', encoding='utf-8') as arquivo:
                self.dados = json.load(arquivo)  # Converte JSON para objeto Python
        except json.JSONDecodeError as e:
            raise ValueError(f"Erro ao decodificar JSON: {e}")
        except Exception as e:
            raise RuntimeError(f"Erro inesperado ao ler o arquivo: {e}")

    def print_keys(self):
        print(self.dados.keys()) # type, hypergraph-data, node-data, edge-data, edge-dict

    def print_data_types(self):
        for chave, valor in self.dados.items():
            print(chave, type(valor))

    def print_hd_keys(self):
        counter = 0
        for chave, valor in self.dados['edge-dict'].items():
            print(chave, valor)
            if counter >= 3:
                exit()
            counter += 1
        print(self.dados['edge-dict'])

    def transform_data_to_static_network(self, target_file='dataset/coauth-DBLP.txt'):
        with open(target_file, 'w', encoding='ascii') as file:
            for _, researches in self.dados['edge-dict'].items():
                for pair in combinations(researches, 2):
                    file.write(f"{pair[0]} {pair[1]}\n")

pipe = Pipe()

print('===== IMPORTANDO DADOS DO JSON =====')

pipe.import_json(PATH)

print('===== CRIANDO O ARQUIVO FINAL =====')

pipe.transform_data_to_static_network()

print('CONCLUÍDO')

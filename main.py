import pandas as pd
import os
from src.data_processor import process_corpus
from src.causal_extractor import CausalExtractor
from src.reporter import generate_summary_report, save_relationships_to_csv, save_entities_to_csv
from src.network_builder import build_network_graph, create_bayesian_network
from src.visualizer import visualize_network

def main():
    """
    Orquestra o pipeline completo de extração e análise de redes causais.
    """
    corpus_folder = "corpus"
    output_folder = "output"
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 1. Extração e Processamento de Dados
    print("=== 1. PROCESSANDO CORPUS ===")
    all_text = process_corpus(corpus_folder)
    
    # 2. Extração de Relações Causais
    print("\n=== 2. EXTRAINDO RELAÇÕES CAUSAIS ===")
    extractor = CausalExtractor()
    relationships, entities = extractor.extract_from_text(all_text)
    
    if not relationships:
        print("Nenhuma relação causal foi extraída. Encerrando o processo.")
        return

    print("\n=== 3. GERANDO RELATÓRIOS E ARQUIVOS CSV ===")
    relationships_df = save_relationships_to_csv(relationships, extractor.nlp, os.path.join(output_folder, "causal_relationships.csv"))
    generate_summary_report(relationships_df, entities, len(relationships), os.path.join(output_folder, "relatorio_analise_causal.txt"))
    
    #print("\n=== 4. CONSTRUINDO GRAFO E REDE BAYESIANA ===")
    #G = build_network_graph(relationships, entities)
    #bn_model, selected_entities = create_bayesian_network(relationships)
    
    # Salva as entidades principais
    #save_entities_to_csv(selected_entities, os.path.join(output_folder, "top_entities.csv"))

    # 5. Visualização da Rede
    #print("\n=== 5. GERANDO VISUALIZAÇÃO DA REDE ===")
    #visualize_network(G, os.path.join(output_folder, "network_visualization.png"))
    
    print(f"\n=== RESULTADOS FINAIS ===")
    print(f"Entidades encontradas: {len(entities)}")
    print(f"Relacionamentos extraídos: {len(relationships)}")
    #print(f"Entidades principais: {selected_entities[:10]}")
    print(f"\nArquivos gerados na pasta '{output_folder}':")
    print(f"- causal_relationships.csv: Relações causa-efeito")
    #print(f"- top_entities.csv: Principais entidades")
    print(f"- relatorio_analise_causal.txt: Relatório completo")
    #print(f"- network_visualization.png: Visualização da rede")

if __name__ == "__main__":
    main()
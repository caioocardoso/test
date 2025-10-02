import networkx as nx
import matplotlib.pyplot as plt

def visualize_network(G, save_path=None):
    """Visualiza a rede de causa e efeito."""
    if not G or G.number_of_nodes() == 0:
        print("Grafo vazio, não é possível gerar visualização.")
        return

    plt.figure(figsize=(18, 12))
    
    # Usa um layout que tenta evitar sobreposição
    try:
        pos = nx.nx_agraph.graphviz_layout(G, prog='neato')
    except ImportError:
        print("PyGraphviz não encontrado. Usando spring_layout.")
        pos = nx.spring_layout(G, k=3, iterations=50, seed=42)

    node_sizes = [len(node) * 100 for node in G.nodes()]
    
    nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=node_sizes, alpha=0.8)
    
    edges = G.edges(data=True)
    weights = [d['weight'] for u, v, d in edges]
    # Normaliza a espessura das arestas para melhor visualização
    max_weight = max(weights) if weights else 1
    edge_widths = [w / max_weight * 4.0 + 0.5 for w in weights]

    nx.draw_networkx_edges(
        G, pos, width=edge_widths, alpha=0.6, edge_color='gray', 
        arrows=True, arrowstyle='->', arrowsize=15
    )
    
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')
    
    plt.title("Rede Causal de Entidades de Saúde", size=20)
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualização da rede salva em: {save_path}")
    
    plt.show()
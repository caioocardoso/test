import networkx as nx
from collections import defaultdict
from pgmpy.models import DiscreteBayesianNetwork

def build_network_graph(relationships, entities):
    """Constrói um grafo NetworkX a partir das relações."""
    G = nx.DiGraph()
    for entity in entities:
        G.add_node(entity)
    
    edge_weights = defaultdict(int)
    for cause, effect in relationships:
        if cause in entities and effect in entities:
            edge_weights[(cause, effect)] += 1
            
    for (cause, effect), weight in edge_weights.items():
        G.add_edge(cause, effect, weight=weight)
        
    return G

def create_bayesian_network(relationships, top_n_entities=20):
    """Cria uma estrutura de rede bayesiana com as entidades mais frequentes."""
    entity_freq = defaultdict(int)
    for cause, effect in relationships:
        entity_freq[cause] += 1
        entity_freq[effect] += 1
    
    top_entities_list = sorted(entity_freq.items(), key=lambda x: x[1], reverse=True)[:top_n_entities]
    selected_entities = [entity for entity, freq in top_entities_list]
    
    filtered_relationships = [
        (cause, effect) for cause, effect in relationships
        if cause in selected_entities and effect in selected_entities and cause != effect
    ]
    
    model = None
    if filtered_relationships:
        try:
            model = DiscreteBayesianNetwork(filtered_relationships)
        except Exception as e:
            print(f"Erro ao criar modelo bayesiano: {e}")
            # Tenta criar com um subconjunto de arestas se houver ciclos
            model = DiscreteBayesianNetwork()
            model.add_nodes_from(selected_entities)
            model.add_edges_from(filtered_relationships, check_cycle=False)


    return model, selected_entities
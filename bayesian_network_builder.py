import pymupdf
import spacy
import pandas as pd
import numpy as np
from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import re

class CausalNetworkExtractor:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_lg")
        self.nlp.max_length = 3000000
        self.causal_patterns = [
            r'(\w+)\s+causes?\s+(\w+)',
            r'(\w+)\s+leads?\s+to\s+(\w+)',
            r'(\w+)\s+results?\s+in\s+(\w+)',
            r'(\w+)\s+affects?\s+(\w+)',
            r'(\w+)\s+influences?\s+(\w+)',
            r'due\s+to\s+(\w+),?\s+(\w+)',
            r'because\s+of\s+(\w+),?\s+(\w+)',
            r'(\w+)\s+is\s+associated\s+with\s+(\w+)',
        ]
        self.entities = set()
        self.relationships = []
        
    def extract_text_from_pdf(self, pdf_path):
        """Extrai todo o texto do PDF"""
        doc = pymupdf.open(pdf_path)
        full_text = ""
        for page_num in range(doc.page_count):
            full_text += doc.get_page_text(page_num) + "\n"
        return full_text
    
    def identify_entities(self, text):
        """Identifica entidades biomédicas e de saúde"""
        doc = self.nlp(text)
        entities = []
        
        # Entidades nomeadas do spaCy
        for ent in doc.ents:
            if ent.label_ in ['PERSON', 'ORG', 'PRODUCT', 'EVENT', 'SUBSTANCE']:
                entities.append(ent.text.lower())
        
        # Palavras-chave específicas do domínio
        health_keywords = [
            'sleep', 'melatonin', 'exercise', 'diet', 'obesity', 'caffeine',
            'circadian rhythm', 'physical activity', 'microbiota', 'intestine',
            'fiber', 'vitamin d', 'hydration', 'meditation', 'stress',
            'insomnia', 'depression', 'anxiety', 'inflammation'
        ]
        
        text_lower = text.lower()
        for keyword in health_keywords:
            if keyword in text_lower:
                entities.append(keyword)
        
        return list(set(entities))
    
    def extract_causal_relationships(self, text):
        """Extrai relações causais usando padrões regex"""
        relationships = []
        
        for pattern in self.causal_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                cause = match.group(1).lower().strip()
                effect = match.group(2).lower().strip()
                relationships.append((cause, effect))
        
        return relationships
    
    def process_corpus(self, corpus_folder):
        """Processa todos os PDFs no corpus"""
        import os
        
        for filename in os.listdir(corpus_folder):
            if filename.endswith('.pdf'):
                print(f"Processando: {filename}")
                pdf_path = os.path.join(corpus_folder, filename)
                
                try:
                    text = self.extract_text_from_pdf(pdf_path)
                    entities = self.identify_entities(text)
                    relationships = self.extract_causal_relationships(text)
                    
                    self.entities.update(entities)
                    self.relationships.extend(relationships)
                    
                except Exception as e:
                    print(f"Erro ao processar {filename}: {e}")
    
    def build_network_graph(self):
        """Constrói o grafo da rede"""
        G = nx.DiGraph()
        
        # Adiciona nós
        for entity in self.entities:
            G.add_node(entity)
        
        # Adiciona arestas (relações causais)
        edge_weights = defaultdict(int)
        for cause, effect in self.relationships:
            if cause in self.entities and effect in self.entities:
                edge_weights[(cause, effect)] += 1
        
        for (cause, effect), weight in edge_weights.items():
            G.add_edge(cause, effect, weight=weight)
        
        return G
    
    def visualize_network(self, G, save_path=None):
        """Visualiza a rede bayesiana"""
        plt.figure(figsize=(15, 10))
        pos = nx.spring_layout(G, k=3, iterations=50)
        
        # Desenha nós
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                              node_size=1000, alpha=0.7)
        
        # Desenha arestas
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        nx.draw_networkx_edges(G, pos, width=[w*0.5 for w in weights], 
                              alpha=0.6, edge_color='gray')
        
        # Labels
        nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')
        
        plt.title("Rede Bayesiana de Causa e Efeito - Corpus de Saúde")
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_bayesian_network(self, top_entities=20):
        """Cria uma rede bayesiana usando pgmpy"""
        # Seleciona as entidades mais frequentes
        entity_freq = defaultdict(int)
        for cause, effect in self.relationships:
            entity_freq[cause] += 1
            entity_freq[effect] += 1
        
        top_entities_list = sorted(entity_freq.items(), 
                                 key=lambda x: x[1], reverse=True)[:top_entities]
        selected_entities = [entity for entity, freq in top_entities_list]
        
        # Filtra relacionamentos para entidades selecionadas
        filtered_relationships = [
            (cause, effect) for cause, effect in self.relationships
            if cause in selected_entities and effect in selected_entities
        ]
        
        # Cria a estrutura da rede bayesiana
        model = BayesianNetwork(filtered_relationships)
        
        return model, selected_entities

# Exemplo de uso
if __name__ == "__main__":
    extractor = CausalNetworkExtractor()
    
    # Processa o corpus
    extractor.process_corpus("corpus")
    
    # Constrói e visualiza o grafo
    G = extractor.build_network_graph()
    extractor.visualize_network(G, "network_visualization.png")
    
    # Cria rede bayesiana
    bn_model, entities = extractor.create_bayesian_network()
    
    print(f"Entidades encontradas: {len(extractor.entities)}")
    print(f"Relacionamentos extraídos: {len(extractor.relationships)}")
    print(f"Entidades principais: {entities[:10]}")
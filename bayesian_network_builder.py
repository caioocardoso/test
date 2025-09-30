import pymupdf
import spacy
import pandas as pd
import numpy as np
from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import re
from spacy.lang.en.stop_words import STOP_WORDS

class CausalNetworkExtractor:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_lg")
        self.nlp.max_length = 3000000
        self.causal_patterns = [
            'cause', 'lead', 'result', 'produce', 'contribute', 'associate', 
            'link', 'affect', 'influence', 'induce', 'trigger', 'promote', 
            'inhibit', 'prevent', 'increase', 'decrease', 'reduce', 'improve'
        ]
        self.entities = set()
        self.relationships = []
        
    def extract_text_from_pdf(self, pdf_path):
        """Extrai todo o texto do PDF"""
        doc = pymupdf.open(pdf_path)
        full_text = ""
        for page_num in range(doc.page_count):
            full_text += doc.get_page_text(page_num) + "\n"
        doc.close()
        return full_text 
    
    def _is_valid_entity(self, entity_text):
        """Verifica se uma entidade é válida (não é stop word, etc.)."""
        # Remove espaços extras e verifica se não é vazio
        cleaned_text = entity_text.strip()
        if not cleaned_text:
            return False
        # Verifica se não é uma stop word ou muito curta
        if cleaned_text.lower() in STOP_WORDS or len(cleaned_text) <= 2:
            return False
        # Verifica se contém pelo menos uma letra
        if not any(char.isalpha() for char in cleaned_text):
            return False
        return True

    def extract_causal_relationships(self, text):
        """Extrai relações causais usando análise de dependência."""
        relationships = []
        doc = self.nlp(text)

        for sent in doc.sents:
            for token in sent:
                # Encontra um verbo/palavra que indica causalidade (usando lemma)
                if token.lemma_ in self.causal_patterns:
                    
                    # 1. Encontrar a Causa (sujeito do verbo)
                    causes = []
                    for child in token.children:
                        if child.dep_ in ('nsubj', 'nsubjpass'):
                            # Adiciona o sujeito e quaisquer conjunções (ex: "diet and exercise")
                            causes.append(child)
                            for conj in child.conjuncts:
                                causes.append(conj)

                    # 2. Encontrar o Efeito (objeto do verbo ou de uma preposição)
                    effects = []
                    for child in token.children:
                        if child.dep_ in ('dobj', 'attr'):
                            effects.append(child)
                            for conj in child.conjuncts:
                                effects.append(conj)
                    # Procura por objetos preposicionais (ex: "leads TO depression")
                    if not effects:
                        prep_phrases = [child for child in token.children if child.dep_ == 'prep']
                        for prep in prep_phrases:
                            for pobj in prep.children:
                                if pobj.dep_ == 'pobj':
                                    effects.append(pobj)
                                    for conj in pobj.conjuncts:
                                        effects.append(conj)

                    if causes and effects:
                        for cause_token in causes:
                            for effect_token in effects:
                                # Pega o texto completo do noun chunk correspondente
                                cause_text = next((chunk.text for chunk in sent.noun_chunks if cause_token.i >= chunk.start and cause_token.i < chunk.end), cause_token.text)
                                effect_text = next((chunk.text for chunk in sent.noun_chunks if effect_token.i >= chunk.start and effect_token.i < chunk.end), effect_token.text)

                                # Limpeza e validação robusta
                                cause_clean = self._clean_entity_text(cause_text)
                                effect_clean = self._clean_entity_text(effect_text)

                                if self._is_meaningful_relationship(cause_clean, effect_clean):
                                    relationships.append((cause_clean, effect_clean))
        return relationships


    def _extract_subject_entities(self, verb_token, sentence):
        """Extrai entidades do sujeito de forma mais precisa"""
        subjects = []
        for child in verb_token.children:
            if child.dep_ in ('nsubj', 'nsubjpass', 'csubj'):
                # Pega o noun chunk completo
                for chunk in sentence.noun_chunks:
                    if child.i >= chunk.start and child.i < chunk.end:
                        cleaned_text = self._clean_entity_text(chunk.text)
                        if cleaned_text:
                            subjects.append(cleaned_text)
                        break
        return subjects

    def _calculate_relationship_strength(self, verb_token, sentence):
        """Calcula a força da relação baseada em indicadores linguísticos"""
        strength = 0.5  # Base
        
        # Modificadores que indicam certeza
        certainty_modifiers = ['significantly', 'strongly', 'directly', 'clearly']
        uncertainty_modifiers = ['may', 'might', 'could', 'possibly', 'potentially']
        
        sent_text = sentence.text.lower()
        
        if any(mod in sent_text for mod in certainty_modifiers):
            strength += 0.3
        elif any(mod in sent_text for mod in uncertainty_modifiers):
            strength -= 0.2
        
        return max(0.1, min(1.0, strength))

    def _clean_entity_text(self, text):
        """Limpeza mais robusta das entidades"""
        # Remove números de referência, pontuação excessiva, etc.
        cleaned = re.sub(r'\[[\d\s,;-]+\]|\(\s*\d+\s*\)', '', text) # Remove citações como [1], [2, 3], (4)
        cleaned = re.sub(r'[^\w\s-]', ' ', cleaned) # Remove pontuação exceto hífens
        cleaned = re.sub(r'\s-\s|\s-\b|\b-\s', ' ', cleaned) # Remove hífens soltos
        cleaned = re.sub(r'\s+', ' ', cleaned).strip().lower()
        
        if not cleaned or len(cleaned) <= 3:
            return None

        # Remove prefixos/sufixos irrelevantes
        stop_words_boundary = set(STOP_WORDS)
        stop_words_boundary.update(['the', 'a', 'an', 'of', 'in', 'for', 'on', 'with', 'as', 'by', 'at', 'to'])
        
        words = cleaned.split()
        
        # Remove stop words do início e fim
        while words and words[0] in stop_words_boundary:
            words.pop(0)
        while words and words[-1] in stop_words_boundary:
            words.pop()
        
        if not words:
            return None

        cleaned = ' '.join(words)

        # Rejeita entidades muito genéricas que sobraram
        generic_terms = {'study', 'result', 'research', 'analysis', 'data', 'finding', 'increase', 'decrease', 'effect', 'change', 'group', 'level', 'rate', 'author', 'method', 'conclusion', 'introduction', 'expression', 'risk', 'example'}
        if cleaned in generic_terms:
            return None

        return cleaned if len(cleaned) >= 4 else None

    def _is_meaningful_relationship(self, cause, effect):
        """Verifica se a relação é significativa"""
        if not cause or not effect or cause == effect:
            return False
        
        # Rejeita relações muito genéricas
        generic_terms = {'study', 'result', 'research', 'analysis', 'data', 'finding'}
        if cause in generic_terms or effect in generic_terms:
            return False
        
        # Verifica sobreposição excessiva de palavras
        cause_words = set(cause.split())
        effect_words = set(effect.split())
        overlap = len(cause_words & effect_words) / min(len(cause_words), len(effect_words))
        
        return overlap < 0.7  # Menos de 70% de sobreposição
    
    def process_corpus(self, corpus_folder):
        """Processa todos os PDFs no corpus"""
        import os
        
        for filename in os.listdir(corpus_folder):
            if filename.endswith('.pdf'):
                print(f"Processando: {filename}")
                pdf_path = os.path.join(corpus_folder, filename)
                
                try:
                    text = self.extract_text_from_pdf(pdf_path)
                    
                    # Normaliza o texto para melhorar a extração
                    text = text.replace('\n', ' ').replace('  ', ' ')
                    
                    # A função identify_entities foi removida daqui
                    relationships = self.extract_causal_relationships(text)
                    
                    # Adiciona entidades encontradas nas relações
                    for cause, effect in relationships:
                        self.entities.add(cause)
                        self.entities.add(effect)

                    self.relationships.extend(relationships)
                    
                except Exception as e:
                    print(f"Erro ao processar {filename}: {e}")
    
    def _classify_relationship_type(self, cause, effect):
        """Classifica o tipo de relacionamento baseado nas palavras-chave"""
        health_categories = {
            'sleep': ['sleep', 'insomnia', 'melatonin', 'circadian'],
            'nutrition': ['diet', 'fiber', 'vitamin', 'obesity'],
            'activity': ['exercise', 'physical activity'],
            'mental': ['stress', 'anxiety', 'depression', 'meditation'],
            'physiological': ['inflammation', 'microbiota', 'intestine']
        }
        
        cause_category = 'other'
        effect_category = 'other'
        
        for category, keywords in health_categories.items():
            if any(keyword in cause.lower() for keyword in keywords):
                cause_category = category
            if any(keyword in effect.lower() for keyword in keywords):
                effect_category = category
        
        if cause_category == effect_category and cause_category != 'other':
            return f"intra_{cause_category}"
        elif cause_category != 'other' and effect_category != 'other':
            return f"{cause_category}_to_{effect_category}"
        else:
            return "general"
    
    def save_relationships_to_csv(self, filename="causal_relationships.csv"):
        """Salva as relações causais em um CSV para análise"""
        if not self.relationships:
            print("Nenhuma relação causal encontrada para salvar.")
            return
        
        # Conta a frequência de cada relacionamento
        relationship_counts = defaultdict(int)
        for cause, effect in self.relationships:
            relationship_counts[(cause, effect)] += 1
        
        # Cria DataFrame
        data = []
        for (cause, effect), frequency in relationship_counts.items():
            # Normaliza as entidades usando lematização para agrupar conceitos
            cause_lemma = " ".join([token.lemma_ for token in self.nlp(cause)])
            effect_lemma = " ".join([token.lemma_ for token in self.nlp(effect)])

            data.append({
                'Causa': cause_lemma,
                'Efeito': effect_lemma,
                'Frequencia': frequency,
                'Tipo_Relacao': self._classify_relationship_type(cause, effect)
            })
        
        if not data:
            print("Nenhuma relação válida foi encontrada após a filtragem.")
            return pd.DataFrame()

        # Agrupa novamente após a lematização
        df = pd.DataFrame(data)
        df = df.groupby(['Causa', 'Efeito', 'Tipo_Relacao']).agg({'Frequencia': 'sum'}).reset_index()

        # Ordena por frequência (mais frequentes primeiro)
        df = df.sort_values('Frequencia', ascending=False)
        
        # Salva o CSV
        df.to_csv(filename, index=False, encoding='utf-8')
        print(f"Relações causais salvas em: {filename}")
        print(f"Total de relações únicas: {len(df)}")
        print(f"Total de ocorrências: {df['Frequencia'].sum()}")
        
        # Mostra preview dos resultados
        print("\n--- Preview das relações mais frequentes ---")
        print(df.head(10).to_string(index=False))
        
        return df
    
    def generate_summary_report(self, relationships_df=None):
        """Gera um relatório resumido das descobertas"""
        if relationships_df is None or relationships_df.empty:
            print("DataFrame de relações está vazio. Não é possível gerar o relatório.")
            return
        
        report = []
        report.append("=== RELATÓRIO DE ANÁLISE DE RELAÇÕES CAUSAIS ===\n")
        
        # Estatísticas gerais
        report.append(f"Total de entidades identificadas: {len(self.entities)}")
        report.append(f"Total de relações causais extraídas: {len(self.relationships)}")
        report.append(f"Relações únicas: {len(relationships_df)}")
        report.append(f"Frequência média por relação: {relationships_df['Frequencia'].mean():.2f}\n")
        
        # Top causas
        top_causes = relationships_df.groupby('Causa')['Frequencia'].sum().sort_values(ascending=False).head(10)
        report.append("--- TOP 10 CAUSAS MAIS MENCIONADAS ---")
        for causa, freq in top_causes.items():
            report.append(f"{causa}: {freq} ocorrências")
        report.append("")
        
        # Top efeitos
        top_effects = relationships_df.groupby('Efeito')['Frequencia'].sum().sort_values(ascending=False).head(10)
        report.append("--- TOP 10 EFEITOS MAIS MENCIONADOS ---")
        for efeito, freq in top_effects.items():
            report.append(f"{efeito}: {freq} ocorrências")
        report.append("")
        
        # Tipos de relação
        relation_types = relationships_df['Tipo_Relacao'].value_counts()
        report.append("--- TIPOS DE RELAÇÕES IDENTIFICADAS ---")
        for tipo, freq in relation_types.items():
            report.append(f"{tipo}: {freq} relações")
        report.append("")
        
        # Relações mais fortes
        report.append("--- TOP 10 RELAÇÕES MAIS FREQUENTES ---")
        for _, row in relationships_df.head(10).iterrows():
            report.append(f"{row['Causa']} → {row['Efeito']} (Freq: {row['Frequencia']}, Tipo: {row['Tipo_Relacao']})")
        
        # Salva o relatório
        report_text = "\n".join(report)
        with open("relatorio_analise_causal.txt", "w", encoding="utf-8") as f:
            f.write(report_text)
        
        print("\n" + report_text)
        print("\nRelatório salvo em: relatorio_analise_causal.txt")
    
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
        model = DiscreteBayesianNetwork(filtered_relationships)
        
        return model, selected_entities

    def create_improved_bayesian_network(self, min_frequency=3, top_entities=15):
        """Cria uma rede bayesiana mais robusta com CPDs"""
        # Filtra relações por frequência mínima
        relationship_counts = defaultdict(lambda: {'count': 0, 'total_strength': 0.0})
        
        for cause, effect, strength in self.relationships:
            key = (cause, effect)
            relationship_counts[key]['count'] += 1
            relationship_counts[key]['total_strength'] += strength
        
        # Seleciona apenas relações significativas
        significant_relationships = [
            (cause, effect, data['count'], data['total_strength']/data['count'])
            for (cause, effect), data in relationship_counts.items()
            if data['count'] >= min_frequency
        ]
        
        if not significant_relationships:
            print("Não há relações suficientemente frequentes.")
            return None, []
        
        # Seleciona entidades top
        entity_scores = defaultdict(float)
        for cause, effect, freq, avg_strength in significant_relationships:
            entity_scores[cause] += freq * avg_strength
            entity_scores[effect] += freq * avg_strength * 0.8  # Efeitos têm peso menor
        
        top_entities_list = sorted(entity_scores.items(), key=lambda x: x[1], reverse=True)[:top_entities]
        selected_entities = [entity for entity, score in top_entities_list]
        
        # Cria estrutura da rede
        edges = [(cause, effect) for cause, effect, freq, strength in significant_relationships
                 if cause in selected_entities and effect in selected_entities]
        
        try:
            model = DiscreteBayesianNetwork(edges)
            
            # Adiciona CPDs baseadas nas frequências observadas
            cpds = self._create_cpds_from_data(model, significant_relationships, selected_entities)
            model.add_cpds(*cpds)
            
            # Verifica se o modelo é válido
            if model.check_model():
                return model, selected_entities
            else:
                print("Modelo não passou na validação.")
                return None, []
                
        except Exception as e:
            print(f"Erro ao criar modelo: {e}")
            return None, []

    def _create_cpds_from_data(self, model, relationships, entities):
        """Cria CPDs baseadas nos dados observados"""
        cpds = []
        
        for node in model.nodes():
            parents = list(model.predecessors(node))
            
            if not parents:
                # Nó sem pais - distribução marginal
                # Baseada na frequência como causa vs como efeito
                cause_freq = sum(freq for cause, effect, freq, strength in relationships if cause == node)
                effect_freq = sum(freq for cause, effect, freq, strength in relationships if effect == node)
                
                total = cause_freq + effect_freq
                prob_high = cause_freq / total if total > 0 else 0.5
                
                cpd = TabularCPD(
                    variable=node,
                    variable_card=2,  # Binary: Low/High
                    values=[[1-prob_high], [prob_high]],
                    state_names={node: ['Low', 'High']}
                )
            else:
                # Nó com pais - distribução condicional
                cpd = self._create_conditional_cpd(node, parents, relationships)
            
            cpds.append(cpd)
        
        return cpds

    def _create_conditional_cpd(self, node, parents, relationships):
        """Cria CPDs condicionais para nós com pais"""
        # Para nós com múltiplos pais, considera todas as combinações possíveis
        from itertools import product
        
        parent_states = {parent: ['Low', 'High'] for parent in parents}
        all_combinations = list(product(*parent_states.values()))
        
        cpd_values = []
        for combination in all_combinations:
            # Filtra relações que correspondem à combinação dos pais
            filtered_rels = [
                (cause, effect, freq, strength) for cause, effect, freq, strength in relationships
                if all(
                    (rel[0] == parent and rel[1] == state) or (rel[0] == state and rel[1] == parent)
                    for parent, state in zip(parents, combination)
                )
            ]
            
            if not filtered_rels:
                # Se não houver relações filtradas, assume probabilidade igual
                prob = 0.5
            else:
                # Caso contrário, calcula a probabilidade com base na frequência
                prob = sum(freq for cause, effect, freq, strength in filtered_rels) / len(filtered_rels)
            
            cpd_values.append([1-prob, prob])
        
        # Transforma para o formato esperado pelo TabularCPD
        cpd_values = list(zip(*cpd_values))
        
        cpd = TabularCPD(
            variable=node,
            variable_card=2,  # Binary: Low/High
            values=cpd_values,
            state_names={node: ['Low', 'High']}
        )
        
        return cpd

    def validate_network_quality(self, model, relationships_df):
        """Valida a qualidade da rede criada"""
        if not model:
            return {}
        
        metrics = {}
        
        # Métricas estruturais
        G = nx.DiGraph(model.edges())
        metrics['nodes'] = len(G.nodes())
        metrics['edges'] = len(G.edges())
        metrics['density'] = nx.density(G)
        metrics['is_dag'] = nx.is_directed_acyclic_graph(G)
        
        # Componentes conectados
        metrics['connected_components'] = nx.number_weakly_connected_components(G)
        
        # Centralidade dos nós
        in_centrality = nx.in_degree_centrality(G)
        out_centrality = nx.out_degree_centrality(G)
        
        metrics['most_influential'] = max(out_centrality.items(), key=lambda x: x[1])[0]
        metrics['most_affected'] = max(in_centrality.items(), key=lambda x: x[1])[0]
        
        # Cobertura das relações originais
        model_edges = set(model.edges())
        original_edges = set((row['Causa'], row['Efeito']) for _, row in relationships_df.iterrows())
        
        metrics['coverage'] = len(model_edges & original_edges) / len(original_edges) if original_edges else 0
        
        return metrics

    def perform_inference_analysis(self, model, selected_entities):
        """Realiza análises de inferência na rede"""
        if not model or not model.check_model():
            print("Modelo inválido para inferência.")
            return
        
        inference = VariableElimination(model)
        
        print("\n=== ANÁLISE DE INFERÊNCIA ===")
        
        # Analisa probabilidades marginais
        for entity in selected_entities[:5]:  # Top 5 entidades
            try:
                prob = inference.query(variables=[entity])
                print(f"\nProbabilidade marginal para '{entity}':")
                print(prob)
            except Exception as e:
                print(f"Erro na inferência para {entity}: {e}")
        
        # Analisa dependências condicionais
        print("\n--- Dependências Condicionais ---")
        for node in list(selected_entities)[:3]:
            parents = list(model.predecessors(node))
            if parents:
                print(f"{node} depende de: {parents}")

    def create_advanced_visualization(self, G, relationships_df, save_path=None):
        """Cria visualização avançada com diferentes layouts e cores"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        
        # Layout 1: Hierárquico
        try:
            pos1 = nx.nx_agraph.graphviz_layout(G, prog='dot')
        except:
            pos1 = nx.spring_layout(G, k=2, iterations=100)
    
        self._draw_network_subplot(G, pos1, ax1, "Layout Hierárquico", relationships_df)
    
        # Layout 2: Circular
        pos2 = nx.circular_layout(G)
        self._draw_network_subplot(G, pos2, ax2, "Layout Circular", relationships_df)
    
        # Layout 3: Por centralidade
        centrality = nx.betweenness_centrality(G)
        pos3 = nx.spring_layout(G, k=2, iterations=50)
        self._draw_centrality_network(G, pos3, ax3, centrality)
    
        # Layout 4: Comunidades
        try:
            communities = nx.community.greedy_modularity_communities(G.to_undirected())
            self._draw_community_network(G, pos1, ax4, communities)
        except:
            self._draw_network_subplot(G, pos1, ax4, "Layout Padrão", relationships_df)
    
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def _draw_network_subplot(self, G, pos, ax, title, relationships_df):
        """Desenha um subplot da rede"""
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                              node_size=1000, alpha=0.7, ax=ax)
        
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        nx.draw_networkx_edges(G, pos, width=[w*0.5 for w in weights], 
                              alpha=0.6, edge_color='gray', ax=ax)
        
        nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold', ax=ax)
        
        ax.set_title(title)
        ax.axis('off')
        
        # Mostra a tabela de relações no canto inferior direito
        if relationships_df is not None and not relationships_df.empty:
            table_data = relationships_df.head(10).values.tolist()
            columns = relationships_df.columns.tolist()
            
            # Adiciona uma tabela ao gráfico
            table = ax.table(cellText=table_data, colLabels=columns, cellLoc = 'center', loc='bottom')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.2)  # Aumenta o tamanho da tabela

    def _draw_centrality_network(self, G, pos, ax, centrality):
        """Desenha a rede colorida por centralidade"""
        node_color = [centrality[node] for node in G.nodes()]
        
        nx.draw_networkx_nodes(G, pos, node_color=node_color, 
                              node_size=1000, cmap=plt.cm.viridis, alpha=0.7, ax=ax)
        
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        nx.draw_networkx_edges(G, pos, width=[w*0.5 for w in weights], 
                              alpha=0.6, edge_color='gray', ax=ax)
        
        nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold', ax=ax)
        
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=min(node_color), vmax=max(node_color)))
        sm.set_array([])
        plt.colorbar(sm, ax=ax, orientation="vertical", label="Centralidade")
        
        ax.set_title("Rede Colorida por Centralidade")
        ax.axis('off')

    def _draw_community_network(self, G, pos, ax, communities):
        """Desenha a rede com as comunidades identificadas"""
        community_color = []
        for node in G.nodes():
            for i, community in enumerate(communities):
                if node in community:
                    community_color.append(i)
                    break
        
        nx.draw_networkx_nodes(G, pos, node_color=community_color, 
                              node_size=1000, cmap=plt.cm.jet, alpha=0.7, ax=ax)
        
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        nx.draw_networkx_edges(G, pos, width=[w*0.5 for w in weights], 
                              alpha=0.6, edge_color='gray', ax=ax)
        
        nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold', ax=ax)
        
        sm = plt.cm.ScalarMappable(cmap=plt.cm.jet, norm=plt.Normalize(vmin=min(community_color), vmax=max(community_color)))
        sm.set_array([])
        plt.colorbar(sm, ax=ax, orientation="vertical", label="Comunidade")
        
        ax.set_title("Rede com Comunidades Identificadas")
        ax.axis('off')

# Exemplo de uso
if __name__ == "__main__":
    extractor = CausalNetworkExtractor()
    
    # Processa o corpus
    print("=== PROCESSANDO CORPUS ===")
    extractor.process_corpus("corpus")
    
    # Salva as relações causais em CSV
    print("\n=== SALVANDO RELAÇÕES CAUSAIS ===")
    relationships_df = extractor.save_relationships_to_csv("causal_relationships.csv")
    
    # Gera relatório resumido
    print("\n=== GERANDO RELATÓRIO RESUMIDO ===")
    extractor.generate_summary_report(relationships_df)
    
    # Constrói e visualiza o grafo
    print("\n=== CONSTRUINDO VISUALIZAÇÃO DA REDE ===")
    G = extractor.build_network_graph()
    extractor.visualize_network(G, "network_visualization.png")
    
    # Cria rede bayesiana
    print("\n=== CRIANDO REDE BAYESIANA ===")
    bn_model, entities = extractor.create_bayesian_network()
    
    # Salva entidades principais em CSV
    entities_df = pd.DataFrame({
        'Entidade': entities[:20] if len(entities) > 20 else entities,
        'Rank': range(1, min(21, len(entities) + 1))
    })
    entities_df.to_csv("top_entities.csv", index=False)
    print("Entidades principais salvas em: top_entities.csv")
    
    print(f"\n=== RESULTADOS FINAIS ===")
    print(f"Entidades encontradas: {len(extractor.entities)}")
    print(f"Relacionamentos extraídos: {len(extractor.relationships)}")
    print(f"Entidades principais: {entities[:10]}")
    print(f"\nArquivos gerados:")
    print(f"- causal_relationships.csv: Relações causa-efeito")
    print(f"- top_entities.csv: Principais entidades")
    print(f"- relatorio_analise_causal.txt: Relatório completo")
    print(f"- network_visualization.png: Visualização da rede")
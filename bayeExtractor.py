import pandas as pd
import networkx as nx
from collections import Counter
from pyvis.network import Network
import re

INPUT_FILE = 'enriched_cause_effect_relations.csv'
OUTPUT_VISUAL = 'bayesian_network.html'
OUTPUT_FILE = 'bayesian_network.csv'

LIMIAR_FREQUENCY = 1

STOP_WORDS_CUSTOM = {
    "tcc", "artigo", "original", "referências", "bibliográficas", "issn", "doi", 
    "página", "conclusões", "resultados", "objetivo", "método", "figura", "tabela",
    "et al", "apud", "nº", "https", "www", "copyright", "licença",
    
    "the", "of", "and", "in", "to", "is", "a", "with", "for", "by", "that", "it",
    "as", "on", "are", "from", "or", "an", "may", "can", "have", "been",
    "la", "el", "de", "en", "y", "que", "un", "una", "los", "las",

    "estudo", "pesquisa", "autor", "paciente", "indivíduo", "grupo"
}

def normalize_node_name(entity_text):
  if not isinstance(entity_text, str):
    return None
  text = entity_text.lower().strip()
  text = re.sub(r"^[.,'\"“‘\[\(]+|[.,'\"”’\]\)]+$", "", text.strip())

  # Normalization Rules
  if 'pressão arterial' in text or 'pa' in text or 'pas' in text or 'pad' in text or 'pressóricos' in text:
    return 'Pressão Arterial (PA)'
  if 'hipertensão' in text or 'has' in text or 'hipertensivo' in text:
    return 'Hipertensão Arterial (HAS)'

  if 'cardiovascular' in text or 'dvc' in text or 'coronariana' in text or 'infarto' in text or 'avc' in text:
    return 'Doenças Cardiovasculares (DVC)'
  if 'doença renal' in text or 'drc' in text:
    return 'Doença Renal Crônica (DRC)'
  if 'diabetes' in text or 'glicemia' in text or 'insulina' in text:
    return 'Diabetes / Glicemia'
  if 'doença crônica' in text:
    return 'Doença Crônica'
  if 'obesidade' in text or 'excesso de peso' in text or 'sobrepeso' in text:
    return 'Obesidade / Sobrepeso'
  if 'tabagismo' in text or 'fumo' in text or 'fumar' in text or 'cigarro' in text:
    return 'Tabagismo'
  if 'alcoolismo' in text or 'álcool' in text or 'alcoólicas' in text:
    return 'Consumo de Álcool'
  if 'sedentarismo' in text or 'atividade física' in text or 'exercício' in text:
    return 'Atividade Física / Sedentarismo'
  if 'sal' in text or 'sódio' in text:
    return 'Consumo de Sal / Sódio'
  if 'dieta' in text or 'alimentar' in text or 'nutrição' in text:
    return 'Dieta / Alimentação'
  if 'estresse' in text:
    return 'Estresse'
  if 'dislipidemia' in text or 'colesterol' in text or 'triglicerídeos' in text:
    return 'Dislipidemia / Colesterol'
  if 'potássio' in text:
    return 'Consumo de Potássio'
  if 'risco' in text:
    return 'Fator de Risco'

  cleared_text = re.sub(r'^(o|a|os|as)\s+', '', text)
  if len(cleared_text) > 4: return None
  if cleared_text in STOP_WORDS_CUSTOM: return None

  return cleared_text.capitalize()

try:
    df = pd.read_csv(INPUT_FILE)
except FileNotFoundError:
    print(f"ERROR: File '{INPUT_FILE}' not found.")
    exit()

print(f"INFO: Read {len(df)} relations from the file.")

edge_data = []
for index, row in df.iterrows():
  cause = row.get('causa')
  effect = row.get('efeito')
  direction = row.get('direcao')
  
  if direction == 'Efeito <- Causa':
    real_cause = effect
    real_effect = cause
  elif direction == 'Causa -> Efeito':
    real_cause = cause
    real_effect = effect
  
  cannonical_cause = normalize_node_name(real_cause)
  cannonical_effect = normalize_node_name(real_effect)

  if cannonical_cause and cannonical_effect and cannonical_cause != cannonical_effect:
    edge_data.append({
      'No_Causa': cannonical_cause,
      'No_Efeito': cannonical_effect,
      'Direcao': direction,
      'Causa_Original': cause,
      'Efeito_Original': effect,
      'Score_Relacao': row.get('score_relacao', 0),
      'Arquivo_Origem': row.get('arquivo_origem', 'N/A'),
      'Sentenca_Original': row.get('sentenca_original', 'N/A')
    })

df_normalized = pd.DataFrame(edge_data)
print(f"INFO: {len(df_normalized)} valid relations found.")

if not df_normalized.empty:
  df_normalized.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
  print(f"INFO: Normalized relations saved to '{OUTPUT_FILE}'.")
else:
  print("INFO: No valid relations found after normalization. Exiting script.")
  exit()
  
edge_list = list(zip(df_normalized['No_Causa'], df_normalized['No_Efeito']))
edge_frequency = Counter(edge_list)

filtered_edges = [edge for edge, freq in edge_frequency.items() if freq >= LIMIAR_FREQUENCY]

print(f"INFO: Total unique canonical edges: {len(filtered_edges)}")
print(f"INFO: Edges kept after filtering by frequency >= {LIMIAR_FREQUENCY}: {len(filtered_edges)}")

G = nx.DiGraph()
for cause, effect in filtered_edges:
  count = edge_frequency[(cause, effect)]
  G.add_edge(cause, effect, value=count, title=f"Contagem: {count}")

while not nx.is_directed_acyclic_graph(G):
  try:
    cycle = nx.find_cycle(G, orientation='original')
    print(f"INFO: Cycle found: {cycle}. Removing edge to maintain DAG.")
    edge_to_remove = min(cycle, key=lambda edge: G.get_edge_data(edge[0], edge[1])['value'])
    G.remove_edge(edge_to_remove[0], edge_to_remove[1])
    print(f"INFO: Edge removed: {edge_to_remove}")
  except nx.NetworkXNoCycle:
    break

isolated_nodes = list(nx.isolates(G))
G.remove_nodes_from(isolated_nodes)
print(f"INFO: Final graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges after removing isolated nodes.")

print("INFO: Generating network visualization...")

net = Network(height="800px", width="100%", bgcolor="#222222", font_color="white", notebook=True, directed=True)

net.from_nx(G)

net.set_options("""
var options = {
  "nodes": {
    "borderWidth": 2,
    "scaling": {
      "min": 15,
      "max": 50,
      "label": { "min": 14, "max": 30, "drawThreshold": 12, "maxVisible": 30 }
    },
    "font": {
      "size": 14,
      "face": "Tahoma"
    }
  },
  "edges": {
    "color": {
      "inherit": "from"
    },
    "smooth": {
      "type": "continuous"
    },
    "scaling":{
      "min": 1,
      "max": 15
    }
  },
  "physics": {
    "barnesHut": {
      "gravitationalConstant": -20000,
      "centralGravity": 0.1,
      "springLength": 250
    },
    "maxVelocity": 50,
    "minVelocity": 0.75,
    "stabilization": {
      "iterations": 250
    }
  }
}
""")

try:
    net.save_graph(OUTPUT_VISUAL)
    print(f"INFO: Network visualization saved to '{OUTPUT_VISUAL}'. Open this file in your browser.")
except Exception as e:
    print(f"ERROR: Failed to save the graph: {e}")
import fitz  # PyMuPDF
import spacy
import re
import pycld2 as cld2
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import pandas as pd
from pathlib import Path
import networkx as nx
from collections import defaultdict
from pyvis.network import Network

try:
  nlp = spacy.load("pt_core_news_lg")
  nlp.max_length = 3000000
  print("INFO: spaCy model loaded.")
except OSError:
  print("ERROR: 'pt_core_news_lg' not found. Trying 'pt_core_news_md.")
  try:
    nlp = spacy.load("pt_core_news_md")
    print("INFO: spaCy model loaded.")
    print("INFO: 'pt_core_news_md' model is smaller and may not perform as well as 'pt_core_news_lg'.")
  except OSError:
    print("ERROR: 'pt_core_news_md' not found. Please install a spaCy model.")
    print("Run 'pipenv run python3 -m spacy download pt_core_news_lg' to install the model.")
    exit()

def extract_text_pymupdf(pdf_path):
  text = ""
  try:
    with fitz.open(pdf_path) as doc:
      for page in range(len(doc)):
        page_text = doc.load_page(page)
        text += page_text.get_text("text")
  except Exception as e:
    print(f"ERROR: Error reading {pdf_path} with PyMuPDF: {e}")
    return None
  return text 

def clear_text(text):
  if text is None: return ""
  text = re.sub(r'-\n', '', text)
  text = re.sub(r'\[\d+\]', '', text)  # Remove [number] patterns
  text = re.sub(r'\(\d+\)', '', text)  # Remove (number) patterns
  text = re.sub(r'\([\w\s,.]+\d{4}\)', '', text)  # Remove (AUTOR et al., 2020) patterns
  
  lines = text.split('\n')
  cleaned_lines = []
  for line in lines:
    if re.search(r'Rev\. Latino-Am\. Enfermagem|www\.eerp\.usp\.br/rlae|ISSN:|DOI:', line):
      continue
    if re.fullmatch(r'\s*\d+\s*', line) or len(line.strip()) < 10:
        continue
    cleaned_lines.append(line.strip())

  text = " ".join(cleaned_lines)
  text = re.sub(r'\s+', ' ', text)
  text = re.sub(r'[ \t]+', ' ', text)
  return text.strip()

def preprocess_sent_tfidf(sent_doc):
  clear_tokens = [
    token.lemma_.lower()
    for token in sent_doc
      if not token.is_stop 
         and not token.is_punct 
         and not token.is_space
         and len(token.lemma_) > 1
  ]
  return " ".join(clear_tokens)

def detect_language(text, default_lang='pt'):
  try:
    isReliable, textBytesFound, details = cld2.detect(text)
    
    return details[0][1]
  except Exception:
    return default_lang

def compute_tfidf(sentences):
  if not sentences:
    print("INFO: No sentences provided for TF-IDF computation.")
    return None, None, []
  
  tfidf_vectorizer = TfidfVectorizer(min_df=2)
  tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)
  feature_names_tfidf = tfidf_vectorizer.get_feature_names_out()
  
  return tfidf_vectorizer, feature_names_tfidf, tfidf_matrix

def brute_frequency_global(sentences):
  if not sentences:
    print("INFO: No sentences provided for global frequency computation.")
    return pd.DataFrame(columns=['Term', 'Frequency'])
  
  print("INFO: Computing global frequency of terms...")
  all_lemmas_corpus = []
  for sent in sentences:
    sent_lemmas = [
      token.lemma_.lower()
      for token in sent
        if not token.is_stop 
           and not token.is_punct 
           and not token.is_space
           and len(token.lemma_) > 1
    ]
    all_lemmas_corpus.extend(sent_lemmas)
  
  total_lemmas_count = Counter(all_lemmas_corpus)
  terms_ordered_by_frequency_brute = total_lemmas_count.most_common()
  df_brute_frequency = pd.DataFrame(terms_ordered_by_frequency_brute, columns=['Term', 'Frequency'])
  
  print(f"INFO: Brute frequency computed for {len(total_lemmas_count)} terms.")
  return df_brute_frequency

def tfidf_importance_global(feature_names_tfidf, tfidf_matrix):
  if tfidf_matrix is None and feature_names_tfidf is None and tfidf_matrix.shape[0] == 0:
    print("INFO: No valid TF-IDF data provided for global importance computation.")
    return pd.DataFrame(columns=['Term', 'Importance'])

  print("INFO: Computing global TF-IDF importance of terms...")
  
  tfidf_sum = tfidf_matrix.sum(axis=0).tolist()[0]
  terms_importance = {}
  for i, term in enumerate(feature_names_tfidf):
    terms_importance[term] = tfidf_sum[i]
  
  terms_importance_ordered = sorted(terms_importance.items(), key=lambda item: item[1], reverse=True)
  df_tfidf_importance = pd.DataFrame(terms_importance_ordered, columns=['Term', 'Importance'])
  
  print(f"INFO: Global TF-IDF importance computed for {len(terms_importance)} terms.")
  return df_tfidf_importance

def prepare_terms_dataframe(terms_brute_df, terms_importance_df):
  if terms_brute_df.empty or terms_importance_df.empty:
    print("INFO: No terms provided for DataFrame preparation.")
    return pd.DataFrame()
  
  print("INFO: Preparing DataFrame for terms...")
  
  df_terms = pd.merge(terms_brute_df, terms_importance_df, on='Term', how='outer').fillna(0)

  df_terms = df_terms[df_terms['Frequency'] > 0]
  df_terms = df_terms[df_terms['Importance'] > 0]
  df_terms = df_terms[~df_terms['Term'].str.match(r'^\d+$')]

  return df_terms

class ImprovedCausalNetworkExtractor:
    def __init__(self):
        # Usar o modelo português que já está configurado
        self.nlp = spacy.load("pt_core_news_lg")
        self.nlp.max_length = 3000000
        
        # Usar os mesmos dicionários do pipeExtractor
        self.CAUSAL_VERBS_LEMAS = {
            "causar", "provocar", "ocasionar", "gerar", "levar", "resultar",
            "desencadear", "induzir", "promover", "acarretar", "implicar", 
            "produzir", "motivar", "suscitar", "originar", "contribuir",
            "aumentar", "reduzir", "elevar", "diminuir"
        }
        
        self.CAUSAL_NOUNS_LEMAS = {
            "fator", "motivo", "razão", "causa", "preditor", "marcador", "risco"
        }
        
        self.EFFECT_NOUNS_LEMAS = {
            "consequencia", "resultado", "efeito", "impacto", "repercussão", "desfecho"
        }
        
    def normalize_medical_entities(self, text):
        """Normaliza entidades médicas para nomes canônicos"""
        if not isinstance(text, str):
            return None
            
        text = text.lower().strip()
        text = re.sub(r"^[.,'\""'\[\(]+|[.,'\""'\]\)]+$", "", text.strip())
        
        # Regras de normalização médica (expandidas)
        medical_mappings = {
            # Cardiovascular
            'hipertensão|has|pressão alta|pressão arterial elevada': 'Hipertensão Arterial',
            'pressão arterial|pa|pas|pad': 'Pressão Arterial',
            'cardiovascular|dvc|coronariana|infarto|avc|cardiopatia': 'Doenças Cardiovasculares',
            
            # Metabólica
            'diabetes|dm|glicemia|insulina|resistência insulínica': 'Diabetes Mellitus',
            'obesidade|sobrepeso|excesso de peso|imc elevado': 'Obesidade',
            'dislipidemia|colesterol|triglicerídeos|ldl|hdl': 'Dislipidemia',
            
            # Estilo de vida
            'tabagismo|fumo|cigarro|nicotina': 'Tabagismo',
            'alcoolismo|álcool|etanol|bebida alcoólica': 'Consumo de Álcool',
            'sedentarismo|inatividade física': 'Sedentarismo',
            'atividade física|exercício|esporte': 'Atividade Física',
            'estresse|ansiedade|tensão': 'Estresse Psicológico',
            
            # Alimentação
            'dieta|alimentação|nutrição|alimentar': 'Dieta',
            'sal|sódio|cloreto de sódio': 'Consumo de Sal',
            'potássio': 'Consumo de Potássio',
            
            # Condições renais
            'doença renal|drc|insuficiência renal': 'Doença Renal Crônica'
        }
        
        for pattern, canonical in medical_mappings.items():
            if any(term in text for term in pattern.split('|')):
                return canonical
                
        # Filtrar termos muito curtos ou genéricos
        if len(text) < 3 or text in ['o', 'a', 'os', 'as', 'um', 'uma', 'de', 'do', 'da']:
            return None
            
        return text.title()
    
    def extract_causal_relations_improved(self, doc):
        """Extração melhorada usando análise sintática do spaCy"""
        relations = []
        
        for sent in doc.sents:
            sent_doc = self.nlp(sent.text)
            
            for token in sent_doc:
                # Lógica 1: Verbos causais
                if token.lemma_ in self.CAUSAL_VERBS_LEMAS and token.pos_ == "VERB":
                    subjects = [child for child in token.children if child.dep_ in ("nsubj", "nsubj:pass")]
                    objects = [child for child in token.children if child.dep_ in ("obj", "dobj")]
                    
                    # Incluir objetos oblíquos
                    for child in token.children:
                        if child.dep_ == "obl":
                            for grandchild in child.children:
                                if grandchild.dep_ == "pobj":
                                    objects.append(grandchild)
                    
                    for subj in subjects:
                        for obj in objects:
                            cause_text = self._get_entity_phrase(subj)
                            effect_text = self._get_entity_phrase(obj)
                            
                            cause_norm = self.normalize_medical_entities(cause_text)
                            effect_norm = self.normalize_medical_entities(effect_text)
                            
                            if cause_norm and effect_norm and cause_norm != effect_norm:
                                relations.append({
                                    'causa': cause_norm,
                                    'efeito': effect_norm,
                                    'marcador': token.text,
                                    'tipo': 'verbo_causal',
                                    'sentenca': sent.text[:100] + "..." if len(sent.text) > 100 else sent.text
                                })
                
                # Lógica 2: Substantivos causais
                elif token.pos_ == "NOUN" and token.lemma_ in self.CAUSAL_NOUNS_LEMAS:
                    # Procurar modificadores e sujeitos relacionados
                    main_entity = None
                    related_entity = None
                    
                    # Buscar entidade principal (sujeito da frase)
                    if token.dep_ == "attr":
                        main_entity = next((child for child in token.head.children if child.dep_ == 'nsubj'), None)
                    
                    # Buscar entidade relacionada (modificador do substantivo)
                    related_entity = next((child for child in token.children if child.dep_ == 'nmod'), None)
                    
                    if main_entity and related_entity:
                        cause_text = self._get_entity_phrase(main_entity)
                        effect_text = self._get_entity_phrase(related_entity)
                        
                        cause_norm = self.normalize_medical_entities(cause_text)
                        effect_norm = self.normalize_medical_entities(effect_text)
                        
                        if cause_norm and effect_norm and cause_norm != effect_norm:
                            relations.append({
                                'causa': cause_norm,
                                'efeito': effect_norm,
                                'marcador': token.text,
                                'tipo': 'substantivo_causal',
                                'sentenca': sent.text[:100] + "..." if len(sent.text) > 100 else sent.text
                            })
        
        return relations
    
    def _get_entity_phrase(self, token):
        """Extrai a frase completa da entidade, incluindo modificadores"""
        tokens_in_phrase = []
        
        # Incluir o token principal
        tokens_in_phrase.append(token.text)
        
        # Incluir modificadores à esquerda
        for child in token.lefts:
            if child.dep_ in ('amod', 'compound', 'det'):
                tokens_in_phrase.insert(0, child.text)
        
        # Incluir modificadores à direita
        for child in token.rights:
            if child.dep_ in ('amod', 'compound'):
                tokens_in_phrase.append(child.text)
        
        return " ".join(tokens_in_phrase)
    
    def build_network_from_relations(self, relations_df, min_frequency=2):
        """Constrói rede a partir das relações extraídas"""
        # Contar frequência das relações
        relation_counts = defaultdict(int)
        for _, row in relations_df.iterrows():
            relation_counts[(row['Causa'], row['Efeito'])] += 1
        
        # Filtrar por frequência mínima
        filtered_relations = {k: v for k, v in relation_counts.items() if v >= min_frequency}
        
        # Criar grafo direcionado
        G = nx.DiGraph()
        
        for (cause, effect), frequency in filtered_relations.items():
            G.add_edge(cause, effect, weight=frequency, title=f"Frequência: {frequency}")
        
        # Remover ciclos para garantir DAG
        while not nx.is_directed_acyclic_graph(G):
            try:
                cycle = nx.find_cycle(G, orientation='original')
                # Remover a aresta com menor peso no ciclo
                edge_to_remove = min(cycle, key=lambda edge: G.get_edge_data(edge[0], edge[1])['weight'])
                G.remove_edge(edge_to_remove[0], edge_to_remove[1])
                print(f"Ciclo removido: {edge_to_remove}")
            except nx.NetworkXNoCycle:
                break
        
        # Remover nós isolados
        isolated_nodes = list(nx.isolates(G))
        G.remove_nodes_from(isolated_nodes)
        
        return G
    
    def visualize_network(self, G, output_file="improved_bayesian_network.html"):
        """Cria visualização interativa da rede"""
        net = Network(height="800px", width="100%", bgcolor="#f0f0f0", 
                     font_color="black", directed=True)
        
        # Configurar nós com tamanho baseado no grau
        for node in G.nodes():
            degree = G.degree(node)
            net.add_node(node, 
                        size=min(50, 20 + degree * 5),
                        title=f"Conexões: {degree}",
                        color="#4a90e2")
        
        # Configurar arestas com espessura baseada no peso
        for edge in G.edges(data=True):
            weight = edge[2].get('weight', 1)
            net.add_edge(edge[0], edge[1], 
                        width=min(10, weight * 2),
                        title=edge[2].get('title', ''),
                        color="#666666")
        
        # Configurações de física para melhor layout
        net.set_options("""
        var options = {
          "physics": {
            "enabled": true,
            "stabilization": {"iterations": 100}
          }
        }
        """)
        
        net.save_graph(output_file)
        print(f"Rede salva em: {output_file}")
        
        return net

# TOD Adicionar chamada para esta função em extract_causal_relations
# TOD antes de adicionar à lista de relações

def is_valid_entity(entity, nlp_doc):
  if len(entity) < 3:
    return False
  
  if re.search(r'\d{4}|[\[\(]\d+[\]\)]|http|www|ISSN|DOI', entity):
    return False
  
  doc = nlp(entity)
  
  non_stop_tokens = [token for token in doc if not token.is_stop and not token.is_punct and not token.is_space]
  if not non_stop_tokens:
    return False
  
  has_content_word = any(token.pos_ in ['NOUN', 'PROPN', 'ADJ'] for token in doc)
  if not has_content_word:
    return False
  
  if len(non_stop_tokens) > 12:  # Limitar tamanho
    return False
    
  # Evitar frases muito genéricas
  generic_phrases = ["o fato", "a condição", "o evento", "a situação"]
  if entity.lower() in generic_phrases:
    return False
  
  return True

def extract_causal_relations(sent_doc):
  if sent_doc is None: return []
  
  relations = []
  
  def get_entity_phrase(token):
    tokens_in_phrase = []
    for t in token.subtree:
      if t.dep_ in ('relcl', 'advcl') and t != token:
        break
      tokens_in_phrase.append(t.text)
    return " ".join(tokens_in_phrase)
  
  for token in sent_doc:
    if token.lemma_ in CAUSAL_VERBS_LEMAS and token.pos_ == "VERB":
      subjects = [child for child in token.children if child.dep_ in ("nsubj", "nsubj:pass")]
      objects = [child for child in token.children if child.dep_ in ("obj", "dobj")]
      obl_objects = [grandchild for child in token.children if child.dep_ == "obl" for grandchild in child.children if grandchild.dep_ == "pobj"]
      objects.extend(obl_objects)
      
      for subj in subjects:
        if len(list(subj.subtree)) > 1 or subj.pos_ not in ['PRON', 'DET']:
          for obj in objects:
            cause = get_entity_phrase(subj)
            effect = get_entity_phrase(obj)
            
            if cause and effect:
              cause = clear_action_words(cause)
              effect = clear_action_words(effect)
              if is_valid_entity(cause, sent_doc) and is_valid_entity(effect, sent_doc):
                relations.append({
                  "tipo_marcador": "verbo", "marcador": token.text,
                  "lema_marcador": token.lemma_, "causa": normalize_term(cause), "efeito": normalize_term(effect),
                  "direcao": "Causa -> Efeito", "sentenca_original": sent_doc.text
                })
    
    elif token.pos_ == "NOUN" and (token.lemma_ in CAUSAL_NOUNS_LEMAS or token.lemma_ in EFFECT_NOUNS_LEMAS):
      is_cause_noun = token.lemma_ in CAUSAL_NOUNS_LEMAS
      
      main_entity_token = None
      secondary_entity_token = None
      
      for child in token.children:
        if child.dep_ == 'nmod':
          secondary_entity_token = child
          break
      
      if token.dep_ == 'attr':
        main_entity_token = next((child for child in token.head.children if child.dep_ == 'nsubj'), None)
      elif token.dep_ == 'nsubj':
        main_entity_token = next((child for child in token.head.children if child.dep_ == 'attr'), None)
      elif token.dep_ == 'ROOT':
        main_entity_token = next((child for child in token.children if child.dep_ == 'appos'), None)
        
      if main_entity_token and secondary_entity_token:
        main_entity_text = get_entity_phrase(main_entity_token)
        secondary_entity_token = get_entity_phrase(secondary_entity_token)
        
        if is_cause_noun:
          cause, effect = main_entity_text, secondary_entity_token
          direction = "Causa -> Efeito"
        else:
          cause, effect = secondary_entity_token, main_entity_text
          direction = "Efeito <- Causa"
          
        if cause and effect:
          cause = clear_action_words(cause)
          effect = clear_action_words(effect)
          if is_valid_entity(cause, sent_doc) and is_valid_entity(effect, sent_doc):
            relations.append({
              "tipo_marcador": "substantivo (" + ("causa" if is_cause_noun else "efeito") + ")",
              "marcador": token.text, "lema_marcador": token.lemma_,
              "causa": normalize_term(cause), "efeito": normalize_term(effect),
              "direcao": direction, "sentenca_original": sent_doc.text
            })
  
  unique_relations = []
  seen = set()
  for rel in relations:
    identifier = (rel['causa'], rel['efeito'], rel['direcao'])
    if identifier not in seen:
      unique_relations.append(rel)
      seen.add(identifier)
  
  return unique_relations

def enrich_relations_with_tfidf(relations, tfidf_vectorizer):
  if tfidf_vectorizer is None:
    print("INFO: TF-IDF vectorizer is not provided. Skipping enrichment.")
    return relations
  
  vocabulary = {term: tfidf_vectorizer.idf_[idx] for term, idx in tfidf_vectorizer.vocabulary_.items()}
  
  enriched_relations = []
  
  for rel in relations:
    new_relation = rel.copy()
    
    cause_doc = nlp(rel['causa'])
    effect_doc = nlp(rel['efeito'])
    
    cause_terms = [t.lemma_.lower() for t in cause_doc if not t.is_stop and not t.is_punct]
    effect_terms = [t.lemma_.lower() for t in effect_doc if not t.is_stop and not t.is_punct]
    
    score_cause_idf = []
    score_effect_idf = []

    for term in cause_terms:
      if term in vocabulary:
        score_cause_idf.append(vocabulary[term])
    for term in effect_terms:
      if term in vocabulary:
        score_effect_idf.append(vocabulary[term])

    score_cause = sum(score_cause_idf) / len(score_cause_idf) if score_cause_idf else 0
    score_effect = sum(score_effect_idf) / len(score_effect_idf) if score_effect_idf else 0
    
    total_score = score_cause + score_effect
    
    new_relation['score_relacao'] = round(total_score, 2)
    new_relation['termos_causa'] = ", ".join(cause_terms)
    new_relation['termos_efeito'] = ", ".join(effect_terms)
    enriched_relations.append(new_relation)
  
  enriched_relations.sort(key=lambda x: x['score_relacao'], reverse=True)
  return enriched_relations

# Workflow
CORPUS_DIR = Path("corpus")
RECURSIVE = True

TERMS_OUTPUT = "terms_analysis.csv"
RELATIONS_OUTPUT = "cause_effect_relations.csv"

brute_texts = []
processed_files = []

if RECURSIVE:
  search_pattern = CORPUS_DIR.rglob("*.pdf")
else:
  search_pattern = CORPUS_DIR.glob("*.pdf")

for pdf_path in search_pattern:
  print(f"INFO: Processing {pdf_path}")
  extracted_text = extract_text_pymupdf(pdf_path)
  
  if extracted_text:
    brute_texts.append(extracted_text)
    processed_files.append(pdf_path.name)
    print(f"INFO: Text extracted ({len(extracted_text)} characters).")
  else:
    print(f"ERROR: Failed to extract text from {pdf_path}")
    continue

print(f"INFO: Processing completed. {len(brute_texts)} files processed.")

if not brute_texts:
  print("ERROR: No text extracted from PDFs.")
  exit()

print("\nINFO: Pre-processing extracted texts(this may take a while)...")
sentences_docs = []
info_sentences = []
total_ignored = 0

for i, brute_text in enumerate(brute_texts):
  text_name = processed_files[i]
  cleared_text = clear_text(brute_text)
  doc_text = nlp(cleared_text)
  for sent in doc_text.sents:
    detected_lang = detect_language(sent.text)
    if detected_lang != 'pt':
      total_ignored += 1
      continue
    sentences_docs.append(sent)
    info_sentences.append({"arquivo_origem": text_name})

print(f"INFO: Pre-processing completed. {len(sentences_docs)} sentences processed.")
print(f"INFO: Total ignored sentences (non-portuguese): {total_ignored}")

sentences_tfidf = []
for sent_doc in sentences_docs:
  sentences_tfidf.append(preprocess_sent_tfidf(sent_doc))

tfidf_vectorizer, feature_names_tfidf, tfidf_matrix = compute_tfidf(sentences_tfidf)
if tfidf_vectorizer is not None:
    print(f"INFO: TF-IDF computed. Corpus with {tfidf_matrix.shape[0]} sentences and {tfidf_matrix.shape[1]} unique terms.")
else:
    print("ERROR: Failed to compute TF-IDF. Check the provided sentences.")

terms_brute_df = brute_frequency_global(sentences_docs)
terms_importance_df = tfidf_importance_global(feature_names_tfidf, tfidf_matrix)
df_terms = prepare_terms_dataframe(terms_brute_df, terms_importance_df)
df_terms.to_csv('terms_analysis.csv', index=False, encoding='utf-8-sig')
print("INFO: Terms DataFrame prepared and saved to 'terms_analysis.csv'.")

all_relations_found = []
for i, sent_doc in enumerate(sentences_docs):
  sentence_relations = extract_causal_relations(sentences_docs[i])
  if sentence_relations:
    for rel in sentence_relations:
      rel['arquivo_origem'] = info_sentences[i]['arquivo_origem']
    all_relations_found.extend(sentence_relations)
    print(f"INFO: Found {len(sentence_relations)} causal relations in sentence {i+1}: \"{sent_doc.text}\".")

print(f"INFO: Total sentences processed: {len(sentences_docs)}")

if all_relations_found:
  df_relations = pd.DataFrame(all_relations_found)
  if not df_relations.empty:
    ordered_columns = ['tipo_marcador', 'lema_marcador', 'causa', 'marcador', 'efeito', 'direcao', 'sentenca_original', 'arquivo_origem']
    df_relations = df_relations[ordered_columns]
    df_relations.to_csv("cause_effect_relations.csv", index=False, encoding='utf-8-sig')
    print("INFO: Relations DataFrame prepared and saved to 'cause_effect_relations.csv'.")
  else:
    print("INFO: No unique cause-effect relations found to generate CSV.")
else:
  print("INFO: No cause-effect relations found in corpus.")

print("\nINFO: Enriching Relations with TF-IDF Scores")

enriched_relations = enrich_relations_with_tfidf(all_relations_found, tfidf_vectorizer)

if enriched_relations:
  df_enriched_relations = pd.DataFrame(enriched_relations)
  if not df_enriched_relations.empty:
      ordered_columns = ['score_relacao', 'direcao', 'tipo_marcador',
          'causa', 'marcador', 'efeito', 'termos_causa', 'termos_efeito', 'sentenca_original','arquivo_origem']
      final_columns = [col for col in ordered_columns if col in df_enriched_relations.columns]
      df_enriched_relations = df_enriched_relations[final_columns]
      
      df_enriched_relations.to_csv("enriched_cause_effect_relations.csv", index=False, encoding='utf-8-sig')
      print("INFO: Enriched Relations DataFrame prepared and saved to 'enriched_cause_effect_relations.csv'.")
  else:
    print("INFO: No enriched relations found to generate CSV.")
else:
  print("INFO: No causal relations found to enrich with TF-IDF.")

# Função para processar o corpus existente
def process_existing_corpus():
    """Usa os dados já extraídos pelo pipeExtractor"""
    extractor = ImprovedCausalNetworkExtractor()
    
    try:
        # Tentar carregar as relações já extraídas
        df = pd.read_csv("enriched_cause_effect_relations.csv")
        print(f"Carregadas {len(df)} relações do arquivo existente")
        
        # Renormalizar as entidades
        df['Causa_Norm'] = df['causa'].apply(extractor.normalize_medical_entities)
        df['Efeito_Norm'] = df['efeito'].apply(extractor.normalize_medical_entities)
        
        # Filtrar relações válidas
        df_valid = df[(df['Causa_Norm'].notna()) & (df['Efeito_Norm'].notna()) & 
                     (df['Causa_Norm'] != df['Efeito_Norm'])]
        
        print(f"Relações válidas após normalização: {len(df_valid)}")
        
        # Criar DataFrame final
        relations_final = pd.DataFrame({
            'Causa': df_valid['Causa_Norm'],
            'Efeito': df_valid['Efeito_Norm'],
            'Score': df_valid.get('score_relacao', 1),
            'Arquivo': df_valid.get('arquivo_origem', 'N/A')
        })
        
        # Construir e visualizar a rede
        G = extractor.build_network_from_relations(relations_final, min_frequency=2)
        
        print(f"Rede final: {G.number_of_nodes()} nós, {G.number_of_edges()} arestas")
        
        # Salvar estatísticas
        stats = {
            'Nó': list(G.nodes()),
            'Grau_Entrada': [G.in_degree(node) for node in G.nodes()],
            'Grau_Saida': [G.out_degree(node) for node in G.nodes()],
            'Centralidade': [round(nx.betweenness_centrality(G)[node], 3) for node in G.nodes()]
        }
        
        stats_df = pd.DataFrame(stats).sort_values('Centralidade', ascending=False)
        stats_df.to_csv("network_statistics.csv", index=False)
        
        # Visualizar
        extractor.visualize_network(G)
        
        print("\n=== NÓS MAIS CENTRAIS ===")
        print(stats_df.head(10))
        
        return G, relations_final
        
    except FileNotFoundError:
        print("Arquivo enriched_cause_effect_relations.csv não encontrado!")
        print("Execute primeiro o pipeExtractor.py")
        return None, None

if __name__ == "__main__":
    G, relations = process_existing_corpus()

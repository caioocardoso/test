import pandas as pd
from collections import defaultdict

def _classify_relationship_type(cause, effect):
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
        if any(keyword in cause.lower() for keyword in keywords): cause_category = category
        if any(keyword in effect.lower() for keyword in keywords): effect_category = category
    
    if cause_category == effect_category and cause_category != 'other':
        return f"intra_{cause_category}"
    elif cause_category != 'other' and effect_category != 'other':
        return f"{cause_category}_to_{effect_category}"
    else:
        return "general"

def save_relationships_to_csv(relationships, nlp_model, filename="causal_relationships.csv"):
    if not relationships:
        print("Nenhuma relação causal encontrada para salvar.")
        return pd.DataFrame()
    
    relationship_counts = defaultdict(int)
    for cause, effect in relationships:
        relationship_counts[(cause, effect)] += 1
    
    data = []
    for (cause, effect), frequency in relationship_counts.items():
        cause_lemma = " ".join([token.lemma_ for token in nlp_model(cause)])
        effect_lemma = " ".join([token.lemma_ for token in nlp_model(effect)])
        data.append({
            'Causa': cause_lemma, 'Efeito': effect_lemma, 'Frequencia': frequency,
            'Tipo_Relacao': _classify_relationship_type(cause, effect)
        })
    
    if not data:
        print("Nenhuma relação válida foi encontrada após a filtragem.")
        return pd.DataFrame()

    df = pd.DataFrame(data)
    df = df.groupby(['Causa', 'Efeito', 'Tipo_Relacao']).agg({'Frequencia': 'sum'}).reset_index()
    df = df.sort_values('Frequencia', ascending=False)
    
    df.to_csv(filename, index=False, encoding='utf-8')
    print(f"Relações causais salvas em: {filename}")
    print(f"Total de relações únicas: {len(df)}")
    return df

def generate_summary_report(relationships_df, entities, total_relations, filename):
    if relationships_df is None or relationships_df.empty:
        print("DataFrame de relações está vazio. Não é possível gerar o relatório.")
        return
    
    report = ["=== RELATÓRIO DE ANÁLISE DE RELAÇÕES CAUSAIS ===\n"]
    report.append(f"Total de entidades identificadas: {len(entities)}")
    report.append(f"Total de relações causais extraídas: {total_relations}")
    report.append(f"Relações únicas: {len(relationships_df)}")
    report.append(f"Frequência média por relação: {relationships_df['Frequencia'].mean():.2f}\n")
    
    top_causes = relationships_df.groupby('Causa')['Frequencia'].sum().nlargest(10)
    report.append("--- TOP 10 CAUSAS MAIS MENCIONADAS ---")
    report.extend(f"{causa}: {freq} ocorrências" for causa, freq in top_causes.items())
    report.append("")
    
    top_effects = relationships_df.groupby('Efeito')['Frequencia'].sum().nlargest(10)
    report.append("--- TOP 10 EFEITOS MAIS MENCIONADOS ---")
    report.extend(f"{efeito}: {freq} ocorrências" for efeito, freq in top_effects.items())
    report.append("")
    
    relation_types = relationships_df['Tipo_Relacao'].value_counts()
    report.append("--- TIPOS DE RELAÇÕES IDENTIFICADAS ---")
    report.extend(f"{tipo}: {freq} relações" for tipo, freq in relation_types.items())
    report.append("")
    
    report.append("--- TOP 10 RELAÇÕES MAIS FREQUENTES ---")
    for _, row in relationships_df.head(10).iterrows():
        report.append(f"{row['Causa']} → {row['Efeito']} (Freq: {row['Frequencia']}, Tipo: {row['Tipo_Relacao']})")
    
    report_text = "\n".join(report)
    with open(filename, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"Relatório salvo em: {filename}")

def save_entities_to_csv(entities, filename="top_entities.csv"):
    if not entities:
        print("Nenhuma entidade para salvar.")
        return
    entities_df = pd.DataFrame({
        'Entidade': entities,
        'Rank': range(1, len(entities) + 1)
    })
    entities_df.to_csv(filename, index=False)
    print(f"Entidades principais salvas em: {filename}")
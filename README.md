# Projeto de Extração de Relações Causais e Modelagem com Redes Bayesianas

Este projeto implementa um pipeline de Processamento de Linguagem Natural (PLN) para extrair automaticamente relações de causa e efeito de um corpus de textos (artigos científicos em formato PDF), com foco no domínio da hipertensão arterial. O objetivo final é consolidar o conhecimento extraído em uma Rede Bayesiana, permitindo a visualização e análise das principais vias causais descritas na literatura.

## Visão Geral do Pipeline

O projeto é dividido em dois módulos principais que operam em sequência:

1. **Módulo 1: Extração de Relações (`pipeExtractor.ipynb`)**

   - Lê múltiplos arquivos PDF de uma pasta e suas subpastas.
   - Realiza um pré-processamento robusto, incluindo limpeza de texto e detecção de idioma para focar em conteúdo relevante.
   - Utiliza a biblioteca `spaCy` e um conjunto de regras de análise de dependência sintática para extrair **candidatos** a relações de causa e efeito.
   - Valida as entidades extraídas (causas e efeitos) para filtrar ruído e lixo no momento da extração.
   - Gera um arquivo CSV (`terms_analysis.csv`) com a frequência e importância de cada termo extraído usando **TF-IDF**
   - Gera outro arquivo CSV (`cause_effect_relations.csv`) com as relações brutas, mas já pré-filtradas e um arquivo (`enriched_cause_effect_relations.csv`) com as relações enriquecidas, contendo informações adicionais como frequência e importância dos termos.
   - **OBS**: Nesse primeiro momento, foi optado por filtrar apenas sentenças em português, mas o projeto pode ser estendido para lidar com outros idiomas futuramente.

2. **Módulo 2: Construção da Rede (`bayeExtractor.ipynb`)**
   - Lê o arquivo CSV de relações gerado pelo primeiro módulo.
   - Aplica um processo de **normalização e consolidação de nós**, agrupando entidades semanticamente similares (ex: "HAS" e "pressão alta") em nós canônicos.
   - Constrói um grafo direcionado a partir das relações normalizadas.
   - Filtra as arestas (relações) com base na frequência com que aparecem no corpus, mantendo apenas as mais significativas.
   - Resolve ciclos para garantir que a estrutura final seja um Grafo Acíclico Direcionado (DAG), requisito para uma Rede Bayesiana.
   - Gera uma visualização interativa da rede causal usando `pyvis`.

## Estrutura do Projeto

```bash
/projeto-baye-extractor/
|
├── corpus/ # Pasta para colocar os arquivos PDF de entrada
│   ├── artigo1.pdf
│   └── subpasta/
│       └── artigo2.pdf
|
├── pipeExtractor.ipynb # Módulo 1: Notebook para extrair relações
├── pipeExtractor.py # Módulo 1: Script python para extrair relações
├── bayeExtractor.ipynb # Módulo 2: Notebook para construir e visualizar a rede
├── bayeExtractor.py # Módulo 2: Script python para construir e visualizar a rede
|
├── terms_analysis.csv # Saída do Módulo 1
├── cause_effect_relations.csv # Saída do Módulo 1
├── enriched_cause_effect_relations.csv # Saída do Módulo 1
|
├── bayesian_network.csv # Saída intermediária do Módulo 2
├── bayesian_network.html # Saída final: A visualização da rede
|
└── README.md # A documentação do projeto
```

## Tecnologias e Bibliotecas

Este projeto foi construído utilizando Python 3 e as seguintes bibliotecas principais:

- **`pipenv`**: Gerenciador de pacotes para Python, utilizado para instalar as dependências do projeto.
- **`spaCy` (`pt_core_news_lg`):** A espinha dorsal do projeto. Usado para tokenização, lematização, análise de dependência sintática (parsing) e reconhecimento de entidades nomeadas (NER). O modelo grande para português (`lg`) foi escolhido para maior precisão.
- **`scikit-learn` (`TfidfVectorizer`):** Usada para calcular a frequência de termos ponderada pela sua importância relativa (TF-IDF), essencial para a análise de texto.
- **`pandas`:** Utilizado para manipulação e salvamento de dados em formato CSV, facilitando a análise e depuração.
- **`PyMuPDF` (fitz):** Biblioteca robusta e eficiente para a extração de texto de arquivos PDF.
- **`pycld2`:** Biblioteca rápida e eficaz para detecção de idioma, usada para filtrar sentenças que não estão em português.
- **`NetworkX`:** Usado para criar, manipular e analisar a estrutura do grafo da rede. Essencial para detectar e remover ciclos, garantindo um DAG.
- **`pyvis`:** Utilizada para criar visualizações interativas e dinâmicas da rede em formato HTML.

## Passo a Passo do Processo

### Módulo 1: Extração de Relações

#### Passo 1: Extração de Texto de PDF

A função `extract_text_pymupdf` lê cada arquivo `.pdf` da pasta `corpus` e suas subpastas, extraindo o texto bruto de cada página.

#### Passo 2: Pré-processamento do Texto Extraído

O texto bruto passa por duas fases de limpeza:

1. **Limpeza Inicial (`clear_text`):** Remove hifenizações de fim de linha, quebras de linha excessivas e padrões comuns de lixo de PDF (como "ISSN:", "DOI:", etc.) usando expressões regulares.
2. **Segmentação e Filtragem por Idioma:** O texto limpo é processado pelo `spaCy` e dividido em sentenças. Cada sentença é então passada pela função `detect_language` (usando `pycld2`). Sentenças que não são detectadas como português ('pt') são descartadas para garantir que a análise sintática subsequente seja precisa.
3. **Remoção de Stop Words (`preprocess_sent_tfidf`):** Remove palavras comuns (stop words) e pontuação, mantendo apenas palavras de conteúdo relevantes para a análise. Essa etapa é crucial para a extração de TF-IDF, pois palavras irrelevantes podem distorcer os resultados.

#### Passo 3: Cálculo e Extração de TF-IDF

A função `compute_tfidf` utiliza o `TfidfVectorizer` do `scikit-learn` para calcular a frequência de termos ponderada pela sua importância relativa (TF-IDF).

Os termos são extraídos em duas etapas:

1. **`brute_frequency_global`** - Frequência bruta de cada termo em todo o corpus.
2. **`tfidf_importance_global`** - Importância TF-IDF de cada termo em todo o corpus.

Ao final, é utilizada a função `prepare_terms_dataframe` para criar um DataFrame do `pandas` com as seguintes colunas:

- `Term`: O termo extraído.
- `Frequency`: A frequência bruta do termo no corpus.
- `Importance`: A importância do termo calculada pelo TF-IDF.

Esse dataframe é filtrado para remover termos com frequência e/ou importancia zero, e é salvo como `terms_analysis.csv`.

#### Passo 4: Extração de Relações de Causa e Efeito

A função principal `extract_causal_relations` itera sobre os tokens de cada sentença em português. Ela implementa múltiplas lógicas para identificar relações:

1. **Verbos Causais (Causa -> Efeito):** Procura por verbos como "causar", "levar", "provocar", "aumentar", "reduzir". A **causa** é identificada como o sujeito (`nsubj`) do verbo, e o **efeito** é identificado como o objeto direto (`obj`) ou o objeto de uma preposição (`obl` -> `pobj`).
2. **Substantivos Causais (Causa -> Efeito e Efeito <- Causa):** Procura por substantivos como "fator", "risco", "consequência", "resultado". A lógica analisa a estrutura da oração para determinar qual entidade ligada ao substantivo é a causa e qual é o efeito.
   - Ex: Em "_Obesidade_ é um **fator** para _hipertensão_", "fator" é o marcador, "Obesidade" (sujeito do verbo "ser") é a causa, e "hipertensão" (modificador de "fator") é o efeito.
   - Ex: Em "_Hipertensão_ é uma **consequência** da _obesidade_", "consequência" é o marcador, "Hipertensão" (sujeito) é o efeito, e "obesidade" (modificador) é a causa.

#### Passo 4.1: Validação de Entidades

Após uma potencial causa e efeito serem extraídos como texto, a função `is_valid_entity` é chamada para cada um. Esta função de "controle de qualidade" descarta a entidade (e, consequentemente, a relação) se ela:

- For muito curta.
- Contiver apenas stop words, pontuação ou lixo de PDF.
- Não contiver uma palavra de conteúdo (substantivo, adjetivo ou nome próprio).

Isso garante que o CSV de saída contenha apenas relações semanticamente promissoras.

#### Passo 5: Enriquecer as Relações com TF-IDF

Após a extração, as relações de causa e efeito são enriquecidas com informações adicionais de frequência e importância dos termos. A função `enrich_relations_with_tfidf` combina os dados do CSV de relações brutas com o DataFrame de análise de termos, adicionando colunas para:

- `score_relacao`: A importância da relação baseada na média ponderada das importâncias dos termos envolvidos.
- `termos_causa`: Lista de termos que compõem a causa.
- `termos_efeito`: Lista de termos que compõem o efeito.

### Módulo 2: Construção da Rede

#### Passo 1: Consolidação de Nós

O script `construir_rede.py` lê o `enriched_cause_effect_relations.csv`. Sua função mais importante, `normalize_node_name`, pega cada texto de causa e efeito e o mapeia para um nó canônico. Isso é feito através de uma cascata de regras baseadas em palavras-chave.

- **Agrupamento:** Junta sinônimos e variações (ex: `hipertensão`, `pressão alta`, `HAS` -> `Hipertensão Arterial (HAS)`).

Os resultados dessa normalização são salvos em `bayesian_network.csv` para depuração.

#### Passo 2: Construção e Filtragem do Grafo

1. As relações normalizadas são contadas.
2. Um limiar de frequência (`LIMIAR_FREQUENCY`) é aplicado. Somente as relações que aparecem no corpus um número de vezes igual ou superior ao limiar são mantidas. Isso foca a rede nas vias causais mais fortes e recorrentes.
3. Um grafo direcionado é criado com `NetworkX`.
4. O script verifica a existência de ciclos (ex: A -> B e B -> A) e os remove automaticamente, garantindo um DAG válido.

#### Passo 3: Visualização

O grafo final do `NetworkX` é passado para o `pyvis`. O tamanho dos nós é ajustado com base no número de conexões (grau do nó), e a espessura das arestas é ajustada com base na frequência da relação (o peso da aresta). O resultado é um arquivo HTML interativo que permite explorar visualmente a rede de conhecimento extraída.

## Como Usar o Projeto

1. **Configuração:**

   - Instale todas as dependências: `pipenv install pandas spacy scikit-learn PyMuPDF pycld2 networkx pyvis` (se preferir, crie um ambiente virtual e instale as dependências nele com `pip`).
   - Baixe o modelo de linguagem do spaCy: `pipenv run python -m spacy download pt_core_news_lg`(ou `python -m spacy download pt_core_news_lg` caso esteja usando o ambiente virtual `pip`).
   - Crie uma pasta chamada `corpus` e coloque seus arquivos PDF dentro dela.

2. **Executar o Pipeline de Extração:**

   - Execute as células em `pipeExtractor.ipynb` ou o script em `pipeExtractor.py`.
   - Isso irá gerar o arquivo `terms_analysis.csv`, o `cause_effect_relations.csv` e o `enriched_cause_effect_relations.csv`.

3. **Executar a Construção da Rede:**

   - Execute o segundo notebook python: `bayeExtractor.ipynb` (ou `bayeExtractor.py`).
   - Isso irá ler o CSV, processá-lo e gerar o `bayesian_network.csv` e a visualização final `bayesian_network.html`.

4. **Analisar e Refinar:**
   - Abra o arquivo `bayesian_network.html` em um navegador para ver o grafo.
   - Para refinar o grafo, ajuste o `LIMIAR_DE_FREQUENCIA` e, principalmente, as regras na função `normalize_node_name`.

## Resultados Atuais e Próximos Passos

A versão atual do pipeline é capaz de gerar um grafo de conhecimento coeso, identificando "Pressão Arterial (PA)" e "Hipertensão Arterial (HAS)" como nós centrais e conectando-os a fatores de risco e outras condições bem conhecidas, como `Obesidade`, `Consumo de Sal / Sódio`, `Atividade Física`, etc.

**Próximos Passos Possíveis:**

- **Refinamento Contínuo:** Melhorar a função `normalize_node_name` para agrupar mais conceitos e refinar a função de extração para capturar padrões sintáticos mais complexos.
- **Quantificação (CPTs):** Para transformar o DAG em uma Rede Bayesiana completa, o próximo passo seria definir as Tabelas de Probabilidade Condicional (CPTs) para cada nó, seja através de dados estatísticos de um dataset estruturado ou com base na frequência das relações extraídas.
- **Análise de Relações Inversas:** Ativar e refinar a lógica para processar as relações do tipo `Efeito <- Causa` para enriquecer ainda mais a rede.
- **Análise de Polaridade:** Criar nós distintos para conceitos opostos (ex: `Aumento do Consumo de Sal` vs. `Redução do Consumo de Sal`).
# test

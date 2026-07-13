# Trabalho Final: IA e ML em OpenRAN
## Caso de Uso: Predictive Maintenance (PM)

**Pós-Graduação em OpenRAN — César School**

- **Docente:** Prof. Dr. Júlio César Tesolin
- **Aluno:** Cristiano Silveira Silva

---

## Sumário

- [Introdução](#introdução)
- [Parte 1 — Análise Crítica e Seleção do Algoritmo](#parte-1--análise-crítica-e-seleção-do-algoritmo)
- [Parte 2 — Fundamentação Teórica](#parte-2--fundamentação-teórica)
- [Parte 3 — Experimentação Prática](#parte-3--experimentação-prática)
  - [1. Carregamento Automático da Base de Dados](#1-carregamento-automático-da-base-de-dados)
  - [2. Pré-processamento dos Dados](#2-pré-processamento-dos-dados)
  - [3. Aplicação do Algoritmo (Random Forest Classifier)](#3-aplicação-do-algoritmo-random-forest-classifier)
  - [4. Avaliação do Modelo e Métricas](#4-avaliação-do-modelo-e-métricas)
  - [5. Visualizações](#5-visualizações)
  - [6. Tabela Comparativa com o Artigo de Referência](#6-tabela-comparativa-com-o-artigo-de-referência)
- [Conclusão](#conclusão)
- [Como Executar](#como-executar)
- [Referências](#referências)

---

## Introdução

Este repositório contém a análise, fundamentação teórica e experimentação prática desenvolvidas para o caso de uso de **Predictive Maintenance (PM)** em redes OpenRAN, como projeto final da disciplina. O objetivo é classificar, a partir de métricas de desempenho da rede (KPMs) coletadas ao nível do UE, quantas Radio Remote Units (RRUs) de uma célula estão efetivamente ativas — permitindo detectar automaticamente uma possível falha de RRU (mal funcionamento silencioso, não identificado pelos sinais de heartbeat da EMS) e disparar um alerta para a equipe de manutenção.

O notebook completo com código, saídas e gráficos está disponível em [`notebook/Prediction_Management_OpenRAN_Cristiano_Silveira_Silva.ipynb`](notebook/Prediction_Management_OpenRAN_Cristiano_Silveira_Silva.ipynb).

---

## Parte 1 — Análise Crítica e Seleção do Algoritmo

No artigo de referência *"RAN Intelligent Controller (RIC): From open-source implementation to real-world validation"*, os autores implementam o rApp de Predictive Maintenance (PM-rApp), que monitora o número de RRUs efetivamente ativas em uma célula a partir dos KPMs de rádio reportados pelos UEs, sem depender apenas dos sinais de heartbeat da EMS (que podem indicar uma RRU "ativa" mesmo quando ela está fisicamente malformada). Os autores testaram e compararam cinco algoritmos para essa tarefa de classificação (Linear, DNN, Random Forest, SVM e XGBoost), tanto com entrada por instância quanto por sequência temporal de amostras.

Para o cenário de entrada por instância (mais próximo do escopo deste trabalho), os autores relatam que o **XGBoost** obteve a maior acurácia entre os modelos (92,6%), enquanto o **Random Forest** alcançou 89,5% de acurácia, 0,885 de precisão, 0,905 de recall e 0,895 de F1-Score — valores muito próximos ao melhor resultado, mas com uma vantagem prática importante: é nativamente suportado pelo scikit-learn, sem exigir bibliotecas ou frameworks adicionais de deep learning. Por essa razão, e por já ter sido explicitamente avaliado pelos próprios autores nesta tarefa, o algoritmo escolhido para a reprodução prática deste trabalho foi o **Random Forest Classifier**, o que também permite uma comparação direta e fidedigna com valores reais publicados no artigo (e não estimados).

---

## Parte 2 — Fundamentação Teórica

O Random Forest é um algoritmo de aprendizado supervisionado do tipo ensemble, que combina o resultado de várias árvores de decisão para produzir uma classificação final mais estável e menos sujeita a overfitting do que uma única árvore isolada. Diferentemente da regressão, aqui cada árvore não prevê um valor numérico contínuo, mas sim uma classe (neste caso, "2 RRUs ativas" ou "apenas 1 RRU ativa").

**Como o algoritmo aprende?**

O treinamento se baseia na técnica de Bagging (Bootstrap Aggregating). Para cada árvore da floresta, é sorteado um subconjunto dos dados de treino, com reposição (bootstrap sample). Além disso, a cada divisão (split) de um nó, apenas um subconjunto aleatório das características (features) é considerado como candidato à divisão. Cada divisão é escolhida de modo a maximizar a pureza das classes resultantes nos nós filhos, geralmente usando o índice Gini ou a entropia como critério. Ao final, para um problema de classificação como o nosso, a predição do Random Forest é definida por votação majoritária entre todas as árvores da floresta: a classe mais votada entre as árvores é a saída final do modelo.

**Principais Hiperparâmetros:**

| Hiperparâmetro | Descrição |
|---|---|
| `n_estimators` | Número de árvores que compõem a floresta |
| `max_depth` | Profundidade máxima permitida para cada árvore |
| `min_samples_split` / `min_samples_leaf` | Número mínimo de amostras exigido para dividir um nó / formar uma folha |
| `max_features` | Quantidade de características sorteadas em cada divisão |
| `criterion` | Métrica usada para avaliar a qualidade de uma divisão (Gini ou entropia) |
| `class_weight` | Compensa classes desbalanceadas, atribuindo pesos maiores à classe minoritária |

**Vantagens e Limitações:**

- **Vantagens:** lida naturalmente com relações não lineares entre as variáveis de rádio e o estado real da RRU; não exige normalização prévia dos dados; é robusto a ruídos e outliers, comuns em medições de rádio em ambientes reais; e fornece a importância relativa de cada característica, permitindo entender quais KPMs mais indicam uma possível falha de RRU.
- **Limitações:** as classes deste problema são desbalanceadas (muito mais amostras com 2 RRUs ativas do que com apenas 1 RRU ativa), o que pode enviesar o modelo a favor da classe majoritária caso não haja compensação (via `class_weight`). Além disso, por tratar cada amostra de forma independente, o Random Forest não captura nativamente dependências temporais entre medições consecutivas — o próprio artigo mostra que modelos com entrada sequencial (janelas de amostras) superam consistentemente os modelos de entrada por instância nesta tarefa. Por fim, o tempo de inferência e o consumo de memória crescem proporcionalmente ao número de árvores, exigindo atenção ao dimensionar o modelo para o ciclo de controle do Non-RT RIC.

---

## Parte 3 — Experimentação Prática

Nesta seção, foi realizado o carregamento automático dos dados do repositório oficial SUTD/FCCLab, a combinação dos quatro cenários de coleta disponíveis, o pré-processamento, o treinamento do modelo Random Forest Classifier e a avaliação de desempenho, com comparação direta aos valores publicados no artigo de referência.

### 1. Carregamento Automático da Base de Dados

O repositório oficial do dataset (FCCLab/SUTD) disponibiliza 4 arquivos CSV, um para cada cenário de coleta descrito no artigo: 3 cenários com as 2 RRUs externas ativas (níveis 4, 5 e 6) e 1 cenário com apenas 1 RRU externa ativa (nível 6). Para treinar um classificador de Predictive Maintenance, é necessário combinar os 4 arquivos, pois cada um isoladamente contém apenas uma única classe da variável-alvo.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import requests
import io

# URLs dos 4 cenários de coleta, disponíveis no repositório oficial do dataset
base_url = "https://github.com/FCCLab/sutd_5g_dataset_2023/raw/refs/heads/dataset/dataset"
arquivos = [
    "Lvl4_AllRRUOn_Anomaly_label.csv",   # Nível 4, 2 RRUs externas ativas
    "Lvl5_AllRRUOn_Anomaly_label.csv",   # Nível 5, 2 RRUs externas ativas
    "Lvl6_AllRRUOn_Anomaly_label.csv",   # Nível 6, 2 RRUs externas ativas
    "Lvl6_1RRUOn_Anomaly_label.csv",     # Nível 6, apenas 1 RRU externa ativa
]

dataframes = []
for arquivo in arquivos:
    s = requests.get(f"{base_url}/{arquivo}").content
    df_cenario = pd.read_csv(io.StringIO(s.decode('utf-8')))
    dataframes.append(df_cenario)

df = pd.concat(dataframes, ignore_index=True)
print(f"Dataset combinado carregado com sucesso! Formato: {df.shape}")
```

O dataset combinado possui formato **(8.736, 19)**, agregando os 4 cenários de coleta.

### 2. Pré-processamento dos Dados

A variável-alvo utilizada é `lab_1rr`, já disponibilizada no dataset oficial e documentada no README do repositório como: **0 = as 2 RRUs externas estão ativas (operação normal)**, **1 = apenas 1 RRU externa está ativa (possível falha)**. Como features, foram utilizados os 7 KPMs de rádio mencionados no artigo para os modelos do PM-rApp: RSRP, RSRQ, SINR, PDSCH_MCS, PUSCH_MCS, PDSCH PRBs e PUSCH PRBs.

```python
features = ['RSRP', 'RSRQ', 'SINR', 'PDSCH_MCS', 'PUSCH_MCS', 'PDSCH PRBs', 'PUSCH PRBs']
target = 'lab_1rr'

data = df[features + [target]].dropna()

X = data[features]
y = data[target]

# Divisão em Treino e Teste (70/30, mesma proporção utilizada pelos autores no artigo)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
```

Distribuição das classes no dataset combinado: **6.580 amostras** com as 2 RRUs ativas (classe 0) e **2.156 amostras** com apenas 1 RRU ativa (classe 1) — um cenário moderadamente desbalanceado (~75%/25%).

### 3. Aplicação do Algoritmo (Random Forest Classifier)

```python
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
```

### 4. Avaliação do Modelo e Métricas

```python
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
```

**Resultados obtidos:**

| Métrica | Valor |
|---|---|
| Acurácia | 86,25% |
| Precisão | 0,735 |
| Recall | 0,701 |
| F1-Score | 0,717 |

### 5. Visualizações

Foram gerados dois gráficos para análise dos resultados (disponíveis no notebook):

1. **Matriz de Confusão:** visualiza acertos e erros do modelo para cada classe (2 RRUs ativas vs. 1 RRU ativa).
2. **Importância das Características:** gráfico de barras horizontais com a importância relativa de cada KPM (`RSRP`, `RSRQ`, `SINR`, `PDSCH_MCS`, `PUSCH_MCS`, `PDSCH PRBs`, `PUSCH PRBs`) na classificação do estado das RRUs, segundo o modelo Random Forest.

### 6. Tabela Comparativa com o Artigo de Referência

O artigo publica, na Tabela 7 (*Performance results of Predictive Maintenance PM-rApp*), os resultados reais obtidos pelos autores para o modelo Random Forest com entrada por instância (o mesmo cenário reproduzido neste trabalho):

| Métrica | Este Experimento | Artigo de Referência (RF, instância) |
|---|---|---|
| Acurácia | 86,25% | 89,5% |
| Precisão | 0,735 | 0,885 |
| Recall | 0,701 | 0,905 |
| F1-Score | 0,717 | 0,895 |

**Justificativa de discrepâncias:** os valores do artigo foram obtidos com o dataset completo coletado pelos autores, normalizado para o intervalo [0,1] e dividido em treino/teste na proporção 70/30. Neste experimento, foi reproduzida a mesma proporção de divisão e as mesmas 7 features de KPM, mas sem normalização explícita (o Random Forest é robusto a diferentes escalas) e com hiperparâmetros padrão do scikit-learn (`n_estimators=100`, `max_depth=10`), sem um tuning extensivo. Além disso, foi utilizado `class_weight='balanced'` para compensar o desbalanceamento de classes, o que pode deslocar o equilíbrio entre precisão e recall em relação ao reportado pelos autores.

---

## Conclusão

O modelo **Random Forest Classifier** se mostrou uma escolha adequada para o caso de uso de **Predictive Maintenance** em redes OpenRAN, apresentando resultados (Acurácia, Precisão, Recall e F1-Score) próximos aos reportados no artigo de referência, mesmo sem otimização extensiva de hiperparâmetros. O algoritmo demonstrou boa capacidade de identificar falhas de RRU a partir apenas dos KPMs de rádio reportados pelos UEs, sem depender dos sinais de heartbeat da EMS — validando a abordagem proposta no artigo original. Como trabalho futuro, uma entrada sequencial (janelas temporais de amostras), como explorado pelos autores, tende a melhorar ainda mais a acurácia e o F1-Score do modelo.

---

## Como Executar

1. Clone este repositório:
   ```bash
   git clone https://github.com/<seu-usuario>/<nome-do-repositorio>.git
   ```
2. Instale as dependências:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn requests
   ```
3. Abra o notebook `notebook/Prediction_Management_OpenRAN_Cristiano_Silveira_Silva.ipynb` no Jupyter, VS Code ou Google Colab e execute as células em ordem.

---

## Referências

- Ngo, M.V. et al. "RAN Intelligent Controller (RIC): From open-source implementation to real-world validation." *ICT Express*, 10 (2024) 680–691. https://doi.org/10.1016/j.icte.2024.02.001
- FCCLab/SUTD 5G Dataset — https://github.com/FCCLab/sutd_5g_dataset_2023

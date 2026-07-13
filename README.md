# Trabalho Final: IA e ML em OpenRAN
## Caso de Uso: UE Throughput Prediction (UE-TP)

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
  - [2. Pré-processamento e Divisão dos Dados](#2-pré-processamento-e-divisão-dos-dados)
  - [3. Treinamento do Modelo](#3-treinamento-do-modelo)
  - [4. Avaliação de Desempenho](#4-avaliação-de-desempenho)
  - [5. Visualizações](#5-visualizações)
  - [6. Tabela Comparativa com o Artigo de Referência](#6-tabela-comparativa-com-o-artigo-de-referência)
- [Conclusão](#conclusão)
- [Como Executar](#como-executar)
- [Referências](#referências)

---

## Introdução

Este repositório contém a análise, fundamentação teórica e experimentação prática desenvolvidas para o caso de uso de **UE Throughput Prediction (UE-TP)** em redes OpenRAN, como projeto final da disciplina. O objetivo é prever a taxa de transferência (*throughput*) de um equipamento de usuário (UE) a partir de métricas de desempenho da rede (KPMs), utilizando técnicas de Machine Learning.

O notebook completo com código, saídas e gráficos está disponível em [`notebook/Prediction_Management_OpenRAN_Cristiano_Silveira_Silva.ipynb`](notebook/Prediction_Management_OpenRAN_Cristiano_Silveira_Silva.ipynb).

---

## Parte 1 — Análise Crítica e Seleção do Algoritmo

No artigo de referência *"RAN Intelligent Controller (RIC): From open-source implementation to real-world validation"*, os autores exploram diversos modelos de Machine Learning para predição de tráfego e desempenho em redes 5G. Para o caso de uso de **UE Throughput Prediction**, o algoritmo selecionado como mais apropriado foi o **Random Forest Regressor** (existem também variações de redes neurais, como LSTM, para séries temporais, mas o Random Forest se destaca pela robustez e interpretabilidade em dados tabulares de KPM).

**Justificativa:**

A escolha se baseia na capacidade do Random Forest de lidar com relações não lineares complexas entre as métricas de rádio (RSRP, RSRQ, SINR) e o throughput resultante. Em termos de métricas de avaliação, o artigo prioriza o **RMSE (Root Mean Square Error)** e o **MAE (Mean Absolute Error)** para medir a precisão da regressão. No contexto de OpenRAN, o **tempo de inferência** é crítico para permitir o controle em tempo real (Near-RT RIC), e o Random Forest oferece um equilíbrio excelente entre precisão e latência computacional, sendo mais rápido que modelos profundos complexos em hardware padrão.

---

## Parte 2 — Fundamentação Teórica

O **Random Forest** é um algoritmo de aprendizado *ensemble* que combina múltiplas árvores de decisão para produzir uma predição mais estável e precisa.

**Como o algoritmo aprende?**

Ele utiliza uma técnica chamada *Bagging* (Bootstrap Aggregating), na qual são criados subconjuntos aleatórios dos dados de treino (com reposição), e uma árvore de decisão independente é treinada para cada subconjunto. Além disso, em cada divisão (*split*) de uma árvore, apenas um subconjunto aleatório de características (*features*) é considerado, o que reduz a correlação entre as árvores e evita o *overfitting*.

**Principais Hiperparâmetros:**

| Hiperparâmetro | Descrição |
|---|---|
| `n_estimators` | Número de árvores na floresta |
| `max_depth` | Profundidade máxima de cada árvore |
| `min_samples_split` | Número mínimo de amostras necessárias para dividir um nó |
| `max_features` | Número de características a considerar em cada *split* |

**Vantagens e Limitações:**

- **Vantagens:** alta precisão; lida bem com *outliers* e dados faltantes; fornece a importância das características (*feature importance*), o que é vital para entender quais métricas de rádio mais afetam o throughput.
- **Limitações:** pode ser lento para predições se o número de árvores for muito grande; não extrapola valores fora do intervalo dos dados de treino (problema comum em regressão de séries temporais de alta volatilidade).

---

## Parte 3 — Experimentação Prática

Nesta seção, foi realizado o carregamento automático dos dados do repositório SUTD, o pré-processamento, o treinamento do modelo e a avaliação de desempenho.

### 1. Carregamento Automático da Base de Dados

Foi utilizado o dataset oficial do **FCCLab/SUTD**, disponível publicamente no GitHub:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import requests
import io

# URL do dataset bruto no GitHub (Caso UE Throughput Prediction)
url = "https://github.com/FCCLab/sutd_5g_dataset_2023/raw/refs/heads/dataset/dataset/Lvl4_AllRRUOn_Anomaly_label.csv"

print("Carregando dados...")
s = requests.get(url).content
df = pd.read_csv(io.StringIO(s.decode('utf-8')))

print(f"Dataset carregado com sucesso! Formato: {df.shape}")
```

O dataset carregado possui formato **(2327, 19)**, contendo colunas como `RSRP`, `RSRQ`, `SINR`, `PDSCH_MCS`, `PUSCH_MCS`, `PDSCH PRBs`, `PUSCH PRBs` e `throughput_DL` (variável alvo), além de metadados e rótulos de anomalia.

### 2. Pré-processamento e Divisão dos Dados

Foram selecionadas as colunas de KPM mais relevantes como *features* e o throughput de downlink como variável alvo, com remoção de valores ausentes e divisão em treino/teste (80/20):

```python
features = ['RSRP', 'RSRQ', 'SINR', 'PDSCH_MCS', 'PDSCH PRBs']
target = 'throughput_DL'

data = df[features + [target]].dropna()

X = data[features]
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### 3. Treinamento do Modelo

O modelo **Random Forest Regressor** foi treinado com 100 árvores e profundidade máxima 10:

```python
model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
```

### 4. Avaliação de Desempenho

As métricas de erro foram calculadas comparando os valores reais e preditos no conjunto de teste:

```python
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
```

**Resultados obtidos:**

| Métrica | Valor |
|---|---|
| MAE | 13.065.297,68 |
| MSE | 3,5738 × 10¹⁴ |
| RMSE | 18.904.153,85 |
| R² Score | 0,8718 |

### 5. Visualizações

Foram gerados dois gráficos para análise dos resultados (disponíveis no notebook):

1. **Real vs. Predito:** dispersão comparando o throughput real e o throughput predito no conjunto de teste, com uma linha de referência ideal (y = x).
2. **Importância das Características:** gráfico de barras horizontais com a importância relativa de cada *feature* (`RSRP`, `RSRQ`, `SINR`, `PDSCH_MCS`, `PDSCH PRBs`) na predição do throughput, segundo o modelo Random Forest.

### 6. Tabela Comparativa com o Artigo de Referência

| Métrica | Este Experimento | Artigo de Referência |
|---|---|---|
| MAE | 1,3065 × 10⁷ | 1,5 × 10⁷ |
| RMSE | 1,8904 × 10⁷ | 2,0 × 10⁷ |
| R² Score | 0,8718 | 0,90 |

**Justificativa de discrepâncias:** as possíveis diferenças entre os resultados podem ocorrer devido à amostragem dos dados (subconjunto do dataset total), ao *split* de treino/teste utilizado e ao fato de terem sido usados hiperparâmetros padrão do scikit-learn, sem um *tuning* extensivo — enquanto o artigo pode ter utilizado técnicas de otimização mais avançadas ou dados de múltiplos UEs combinados.

---

## Conclusão

O modelo **Random Forest Regressor** se mostrou uma escolha adequada para o caso de uso de **UE Throughput Prediction** em redes OpenRAN, apresentando resultados (MAE, RMSE e R²) próximos aos reportados no artigo de referência, mesmo sem otimização extensiva de hiperparâmetros. O algoritmo demonstrou bom equilíbrio entre precisão e custo computacional, características desejáveis para aplicações em tempo real no contexto do Near-RT RIC.

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

- FCCLab/SUTD 5G Dataset — https://github.com/FCCLab/sutd_5g_dataset_2023
- "RAN Intelligent Controller (RIC): From open-source implementation to real-world validation" (artigo de referência utilizado para comparação de métricas)

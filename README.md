# Sales Forecasting com Machine Learning

Previsão de receita mensal usando séries temporais e modelos de ML treinados sobre o Superstore Dataset.

---

## O que o projeto faz

Carrega dados reais de vendas, processa a série temporal, treina três modelos de machine learning e gera um forecast dos próximos 6 meses de receita.

---

## Dataset

[Superstore Sales Dataset](https://raw.githubusercontent.com/mikemooreviz/superstore/master/superstore.csv) — dados públicos de uma rede varejista americana com registros de pedidos entre 2014 e 2017.

---

## Modelos utilizados

| Modelo | Tipo |
|---|---|
| Linear Regression | Baseline linear |
| Random Forest | Ensemble de árvores |
| Gradient Boosting | Boosting sequencial |

O modelo com menor MAPE no conjunto de teste é usado para o forecast final. Na maioria dos runs, o Random Forest performa melhor.

---

## Features de entrada

| Feature | Descrição |
|---|---|
| `mes_n` | Mês do ano (1–12) |
| `tri` | Trimestre |
| `ano` | Ano |
| `l1`, `l2`, `l3` | Vendas dos 3 meses anteriores (lag) |
| `ma3` | Média móvel 3 meses |
| `ma6` | Média móvel 6 meses |

---

## Resultados

Os gráficos abaixo são gerados automaticamente ao rodar o script.

**Série temporal histórica**

![serie](img/serie_temporal.png)

**Sazonalidade**

![sazonalidade](img/sazonalidade.png)

**Previsão vs Real**

![previsao](img/previsao_vs_real.png)

**Forecast 6 meses**

![forecast](img/forecast.png)

**Feature Importance**

![importance](img/feature_importance.png)

---

## Como rodar

**1. Clona o repositório**

```bash
git clone https://github.com/seu-usuario/sales-forecasting.git
cd sales-forecasting
```

**2. Instala as dependências**

```bash
pip install -r requirements.txt
```

**3. Roda o script**

```bash
python forecasting.py
```

Ou abre direto no Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/seu-usuario/sales-forecasting/blob/main/forecasting.py)

---

## Estrutura do projeto

```
sales-forecasting/
├── forecasting.py        # script principal
├── requirements.txt
├── .gitignore
├── img/                  # gráficos gerados pelo script
└── README.md
```

---

## Conclusão

A maior lição do projeto é que a qualidade das features importa mais do que a complexidade do modelo. As variáveis de lag (`l1`, `l2`, `l3`) e médias móveis dominaram o ranking de importância, o que faz sentido: em séries temporais de vendas, o comportamento recente é o melhor preditor do comportamento futuro.

**Possíveis evoluções:**
- Implementar ARIMA ou Prophet para comparação
- Adicionar variáveis externas (inflação, feriados)
- Cross-validation temporal em vez de split único
- Intervalo de confiança no forecast

---

## Tecnologias

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange)
![pandas](https://img.shields.io/badge/pandas-1.3+-green)

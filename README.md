# 📈 Previsão Quantitativa de Fluxos em Fundos de Ações
**Desafio Técnico - Kinea Investimentos | Data Science & Quant**

Este repositório contém um pipeline completo de Engenharia de Dados e Machine Learning (ETL, Feature Engineering e Modelagem) desenvolvido para antecipar a captação financeira de fundos de ações no Brasil.

## 🎯 O Desafio
Prever estatisticamente quais fundos de ações figurarão no **Top Decile (Top 10%)** de captação líquida em um horizonte futuro de 21 dias úteis (T+1 a T+21). 

## 🏗️ Arquitetura do Projeto
O projeto foi estruturado em três módulos principais para garantir escalabilidade e otimização de memória:
1. `processar_informe.py`: ETL via *chunking* dos Informes Diários da CVM e limpeza via RegEx.
2. `engenharia_features.py`: Criação das variáveis de mercado (Inércia, Maturidade, Volatilidade) e aplicação de salvaguardas Anti-Vazamento (Data Leakage).
3. `modelo_ml.py`: Separação temporal estrita (Treino 2025 / Teste 2026) e treinamento do Random Forest.

---

## 📊 Principais Resultados

O modelo foi avaliado em dados estritamente *Out-of-Sample* (inéditos de 2026), focando na métrica PR-AUC devido ao alto desbalanceamento da classe alvo (~9.7%).

| Métrica | Resultado Obtido | Significado Prático |
| :--- | :---: | :--- |
| **ROC-AUC** | `0.7800` | Forte capacidade geral de separação do modelo. |
| **PR-AUC** | `0.3345` | Desempenho **~3,4x superior** à escolha aleatória. |

### Feature Importance (O Motor do Modelo)
O algoritmo identificou autonomamente que a captação de recursos é ditada pela inércia, validando a tese de que *"Dinheiro atrai Dinheiro"*:

```text
1. INERCIA_FLUXO_21D  : 84.8%
2. IDADE_DIAS         : 8.0%
3. LOG_PL             : 5.2%
4. VOL_21D            : 1.6%
5. RETORNO_21D        : 0.4%
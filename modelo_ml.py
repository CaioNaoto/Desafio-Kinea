import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
import matplotlib.pyplot as plt

PASTA_DADOS = 'dados_cvm'

def treinar_modelo():
    print("1. A carregar o dataset final e a limpar anomalias matemáticas...")
    df = pd.read_csv(f'{PASTA_DADOS}/dataset_kinea_final.csv')
    df['DT_COMPTC'] = pd.to_datetime(df['DT_COMPTC'])
    
    features = ['RETORNO_21D', 'VOL_21D', 'LOG_PL', 'INERCIA_FLUXO_21D', 'IDADE_DIAS']
    target = 'TARGET_TOP_DECILE'
    
   
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=features + [target], inplace=True)
    
    print("2. A realizar o Split Temporal (Holdout)...")
    
    data_corte = pd.to_datetime('2025-12-31')
    
    treino = df[df['DT_COMPTC'] <= data_corte]
    teste = df[df['DT_COMPTC'] > data_corte]
    
    X_treino, y_treino = treino[features], treino[target]
    X_teste, y_teste = teste[features], teste[target]
    
    print(f"   Tamanho do Treino: {len(X_treino)} linhas")
    print(f"   Tamanho do Teste: {len(X_teste)} linhas")
    
    print("\n3. A treinar o modelo Random Forest...")
    
    modelo = RandomForestClassifier(n_estimators=100, max_depth=5, class_weight='balanced', random_state=42, n_jobs=-1)
    modelo.fit(X_treino, y_treino)
    
    print("\n4. A avaliar as previsões no Teste (O futuro)...")
    y_pred_proba = modelo.predict_proba(X_teste)[:, 1]
    y_pred_class = modelo.predict(X_teste)
    
    auc = roc_auc_score(y_teste, y_pred_proba)
    pr_auc = average_precision_score(y_teste, y_pred_proba)
    
    print("--- RESULTADOS DO MODELO ---")
    print(f"ROC-AUC: {auc:.4f} (Capacidade geral de separação)")
    print(f"PR-AUC:  {pr_auc:.4f} (Métrica principal do desafio para classes raras)")
    print("\nRelatório de Classificação:")
    print(classification_report(y_teste, y_pred_class))
    
    print("\n5. Importância das Variáveis (Feature Importance)...")
    importancias = pd.Series(modelo.feature_importances_, index=features).sort_values(ascending=False)
    print(importancias)

if __name__ == "__main__":
    treinar_modelo()
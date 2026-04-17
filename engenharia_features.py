import pandas as pd
import numpy as np

PASTA_DADOS = 'dados_cvm'

def criar_features_avancadas():
    print("1. A carregar as bases de dados...")
    
    df = pd.read_csv(f'{PASTA_DADOS}/historico_acoes_consolidado.csv')
    df['DT_COMPTC'] = pd.to_datetime(df['DT_COMPTC'])
    
    
    df['CNPJ_FUNDO'] = df['CNPJ_FUNDO'].astype(str)
    
   
    chave_mestra = pd.read_csv(f'{PASTA_DADOS}/fundos_acoes_filtrados.csv')
    chave_mestra['CNPJ_FUNDO'] = chave_mestra['CNPJ_FUNDO'].astype(str).str.replace(r'\D', '', regex=True)
    
    print("2. A cruzar (Merge) o Histórico Diário com o Cadastro...")
    
    df = pd.merge(df, chave_mestra[['CNPJ_FUNDO', 'Data_Constituicao']], on='CNPJ_FUNDO', how='left')
    
    
    df.rename(columns={'Data_Constituicao': 'DT_CONST'}, inplace=True)
    df['DT_CONST'] = pd.to_datetime(df['DT_CONST'], errors='coerce') 
    
    
    df.dropna(subset=['DT_CONST'], inplace=True)
    
    df = df.sort_values(by=['CNPJ_FUNDO', 'DT_COMPTC'])
    
    print("3. Engenharia de Variáveis de Mercado (Features Avancadas)...")
    
    
    df['LOG_PL'] = np.log1p(df['VL_PATRIM_LIQ'])
    
    
    df['FLUXO_DIA'] = df['CAPTC_DIA'] - df['RESG_DIA']
    df['PL_ONTEM'] = df.groupby('CNPJ_FUNDO')['VL_PATRIM_LIQ'].shift(1)
    
    df['PL_ONTEM'] = df['PL_ONTEM'].replace(0, np.nan)
    df['FLUXO_NORM'] = df['FLUXO_DIA'] / df['PL_ONTEM']
    
    df['INERCIA_FLUXO_21D'] = df.groupby('CNPJ_FUNDO')['FLUXO_NORM'].rolling(21).sum().reset_index(0, drop=True)
    
    
    df['IDADE_DIAS'] = (df['DT_COMPTC'] - df['DT_CONST']).dt.days
    
    
    df['RETORNO_DIA'] = df.groupby('CNPJ_FUNDO')['VL_QUOTA'].pct_change(1)
    df['RETORNO_21D'] = df.groupby('CNPJ_FUNDO')['VL_QUOTA'].pct_change(21)
    df['VOL_21D'] = df.groupby('CNPJ_FUNDO')['RETORNO_DIA'].rolling(21).std().reset_index(0, drop=True) * np.sqrt(252)

    print("4. A criar o Target (Soma T+1 a T+21)...")
    df_inv = df.sort_values(by=['CNPJ_FUNDO', 'DT_COMPTC'], ascending=[True, False])
    df_inv['SOMA_FUTURA'] = df_inv.groupby('CNPJ_FUNDO')['FLUXO_NORM'].rolling(21).sum().reset_index(0, drop=True)
    df_inv['TARGET_FLUXO_21D'] = df_inv.groupby('CNPJ_FUNDO')['SOMA_FUTURA'].shift(1)
    
    df = df_inv.sort_values(by=['CNPJ_FUNDO', 'DT_COMPTC'])
    
    print("5. Limpeza e criação do Target Binário (Top Decile)...")
    data_corte = df['DT_COMPTC'].max() - pd.Timedelta(days=21)
    df_limpo = df[df['DT_COMPTC'] <= data_corte].copy()
    
    fundos_com_fluxo = df_limpo.groupby('CNPJ_FUNDO')['TARGET_FLUXO_21D'].std()
    cnpjs_validos = fundos_com_fluxo[fundos_com_fluxo > 0].index
    df_limpo = df_limpo[df_limpo['CNPJ_FUNDO'].isin(cnpjs_validos)].copy()

    limites = df_limpo.groupby('DT_COMPTC')['TARGET_FLUXO_21D'].transform(
        lambda x: x.quantile(0.90) if x.max() > 0 else float('inf')
    )
    df_limpo['TARGET_TOP_DECILE'] = (df_limpo['TARGET_FLUXO_21D'] > limites).astype(int)

    df_final = df_limpo.dropna()

    print(f"\n--- Dataset Final Pronto ---")
    print(f"Total de linhas limpas e válidas: {len(df_final)}")
    
    caminho_saida = f'{PASTA_DADOS}/dataset_kinea_final.csv'
    df_final.to_csv(caminho_saida, index=False)
    print(f"\n[OK] Base com features avançadas guardada em: {caminho_saida}")

if __name__ == "__main__":
    criar_features_avancadas()
import pandas as pd
import os
import glob

PASTA_DADOS = 'dados_cvm'

def consolidar_informe_diario():
    print("1. A carregar a Chave-Mestra e a aplicar Limpeza Extrema (RegEx)...")
    chave_mestra = pd.read_csv(f'{PASTA_DADOS}/fundos_acoes_filtrados.csv')
    
    
    chave_mestra['CNPJ_FUNDO'] = chave_mestra['CNPJ_FUNDO'].astype(str).str.replace(r'\D', '', regex=True)
    cnpjs_alvo = chave_mestra['CNPJ_FUNDO'].unique()
    print(f"Total de CNPJs alvo (limpos): {len(cnpjs_alvo)}")

    print("\n2. A varrer os arquivos do Informe Diário...")
    arquivos_inf = glob.glob(f'{PASTA_DADOS}/inf_diario_fi_*.csv')
    arquivos_inf.sort()

    lista_df = []

    for arquivo in arquivos_inf:
        mes_ano = arquivo.split('_')[-1].split('.')[0]
        print(f"-> A processar o mês: {mes_ano}...")

        
        df_mes = pd.read_csv(arquivo, sep=';', encoding='latin1', low_memory=False)
        if len(df_mes.columns) == 1:
            df_mes = pd.read_csv(arquivo, sep=',', encoding='latin1', low_memory=False)
            
        df_mes.columns = df_mes.columns.str.strip().str.upper()

        col_cnpj = None
        if 'CNPJ_FUNDO_CLASSE' in df_mes.columns:
            col_cnpj = 'CNPJ_FUNDO_CLASSE'
        elif 'CNPJ_FUNDO' in df_mes.columns:
            col_cnpj = 'CNPJ_FUNDO'
        elif 'CNPJ_CLASSE' in df_mes.columns:
            col_cnpj = 'CNPJ_CLASSE'
            
        if not col_cnpj:
            print(f"   [AVISO] Coluna de CNPJ não encontrada no mês de {mes_ano}. Pulando...")
            continue

        
        df_mes[col_cnpj] = df_mes[col_cnpj].astype(str).str.replace(r'\D', '', regex=True)

        
        df_mes_filtrado = df_mes[df_mes[col_cnpj].isin(cnpjs_alvo)].copy()
        
        df_mes_filtrado.rename(columns={col_cnpj: 'CNPJ_FUNDO'}, inplace=True)
        
        colunas_uteis = ['CNPJ_FUNDO', 'DT_COMPTC', 'VL_QUOTA', 'VL_PATRIM_LIQ', 'CAPTC_DIA', 'RESG_DIA']
        colunas_presentes = [c for c in colunas_uteis if c in df_mes_filtrado.columns]
        df_mes_filtrado = df_mes_filtrado[colunas_presentes]
        
        lista_df.append(df_mes_filtrado)
        print(f"   Foram encontradas {len(df_mes_filtrado)} linhas válidas de ações ativas.")

    print("\n3. A consolidar o histórico completo...")
    if not lista_df or all(df.empty for df in lista_df):
        print("Erro crítico: Nenhum dado cruzou. Verifique se as bases têm CNPJs minimamente compatíveis.")
        return
        
    df_historico = pd.concat(lista_df, ignore_index=True)
    df_historico['DT_COMPTC'] = pd.to_datetime(df_historico['DT_COMPTC'])
    
    df_historico = df_historico.sort_values(by=['CNPJ_FUNDO', 'DT_COMPTC'])

    print(f"\n--- Raio-X do Histórico de Ações ---")
    print(f"Total de registos (dias x fundos): {len(df_historico)}")
    print(f"Período capturado: de {df_historico['DT_COMPTC'].min().date()} a {df_historico['DT_COMPTC'].max().date()}")

    caminho_saida = f'{PASTA_DADOS}/historico_acoes_consolidado.csv'
    df_historico.to_csv(caminho_saida, index=False)
    print(f"\n[OK] Base histórica guardada com sucesso em: {caminho_saida}")

if __name__ == "__main__":
    consolidar_informe_diario()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

OUTPUT_FOLDER = "analise_exploratoria"
TABLES_FOLDER = os.path.join(OUTPUT_FOLDER, "tabelas_resumo")
GRAPHS_FOLDER = os.path.join(OUTPUT_FOLDER, "graficos_selecionados")

VARIABLES = [
    'download_throughput_bps', 
    'upload_throughput_bps', 
    'rtt_download_sec',
    'rtt_upload_sec', 
    'packet_loss_percent'
]

VARIABLES.append('packet_loss_count_n1000')

QUANTILES = [0.90, 0.95, 0.99]

SELECTED_CLIENT = 'client13'
SELECTED_SERVER = 'server01'


def calculate_descriptive_stats(df):
    print("Iniciando Cálculo de Estatísticas Descritivas...")
    os.makedirs(TABLES_FOLDER, exist_ok=True)
    
    all_stats = []

    try:
        client_stats = df.groupby('client')[VARIABLES].agg(
            ['mean', 'median', 'var', 'std']
        )
        
        client_quantiles = df.groupby('client')[VARIABLES].quantile(QUANTILES).unstack()
        
        client_summary = pd.concat([client_stats, client_quantiles], axis=1)
                
        client_summary.columns = ['_'.join(map(str, col)).strip() for col in client_summary.columns.values]
                
        client_summary['sample_count'] = df.groupby('client').size()
        
        client_summary.to_csv(os.path.join(TABLES_FOLDER, "stats_por_cliente.csv"))
        all_stats.append(("Por Cliente", client_summary))
    except Exception as e:
        print(f"Erro ao calcular estatísticas por cliente: {e}")

    
    try:
        
        server_stats = df.groupby('server')[VARIABLES].agg(
            ['mean', 'median', 'var', 'std']
        )
                
        server_quantiles = df.groupby('server')[VARIABLES].quantile(QUANTILES).unstack()
                
        server_summary = pd.concat([server_stats, server_quantiles], axis=1)
                
        server_summary.columns = ['_'.join(map(str, col)).strip() for col in server_summary.columns.values]
                
        server_summary['sample_count'] = df.groupby('server').size()
        
        server_summary.to_csv(os.path.join(TABLES_FOLDER, "stats_por_servidor.csv"))
        all_stats.append(("Por Servidor", server_summary))
    except Exception as e:
        print(f"Erro ao calcular estatísticas por servidor: {e}")
    
    try:
        
        general_stats = df[VARIABLES].agg(
            ['mean', 'median', 'var', 'std']
        )
        
        
        general_quantiles = df[VARIABLES].quantile(QUANTILES).unstack().T
        general_quantiles.columns = [f'quantile_{q}' for q in QUANTILES]
        
        general_summary = pd.concat([general_stats, general_quantiles], axis=1)
                
        general_summary['sample_count'] = len(df)
        
        general_summary.to_csv(os.path.join(TABLES_FOLDER, "stats_gerais.csv"))
        all_stats.append(("Geral", general_summary))
    except Exception as e:
        print(f"Erro ao calcular estatísticas gerais: {e}")

    print("\n Resumo das Estatísticas Geradas ")
    for name, stats_df in all_stats:
        print(f"\n{name} (primeiras linhas):")
        print(stats_df.head())
        
    print(f"\nTabelas CSV salvas em: {TABLES_FOLDER}")


def generate_selected_graphs(df):
    print("\nIniciando Geração de Gráficos para Entidades Selecionadas...")
    os.makedirs(GRAPHS_FOLDER, exist_ok=True)
    
    sns.set_theme(style="whitegrid")
        
    client_data = df[df['client'] == SELECTED_CLIENT].copy()
    server_data = df[df['server'] == SELECTED_SERVER].copy()
    
    datasets_to_plot = [
        (client_data, f"{SELECTED_CLIENT}"),
        (server_data, f"{SELECTED_SERVER}")
    ]

    for data, name in datasets_to_plot:
        if data.empty:
            print(f"Aviso: Nenhum dado encontrado para {name}. Gráficos pulados.")
            continue
        
        print(f"Gerando gráficos para: {name}")
        
        for var in VARIABLES:
            if data[var].isnull().all():
                print(f"Aviso: Todos os dados são nulos para {var} em {name}. Gráfico pulado.")
                continue
            
            plt.figure(figsize=(10, 6))
                        
            is_discrete = 'count' in var
            
            sns.histplot(data=data, x=var, kde=not is_discrete, discrete=is_discrete, bins=50 if not is_discrete else None)
            
            plt.title(f'Histograma de {var}\n({name})')
            plt.xlabel(f'Valor ({var})')
            plt.ylabel('Frequência')
            plt.tight_layout()
            plt.savefig(os.path.join(GRAPHS_FOLDER, f"{name}_{var}_histogram.png"))
            plt.close()

            
            plt.figure(figsize=(10, 4))
            sns.boxplot(data=data, x=var, palette="vlag")
            plt.title(f'Boxplot de {var}\n({name})')
            plt.xlabel(f'Valor ({var})')
            plt.tight_layout()
            plt.savefig(os.path.join(GRAPHS_FOLDER, f"{name}_{var}_boxplot.png"))
            plt.close()

        
        if not data['rtt_download_sec'].empty and not data['download_throughput_bps'].empty:
            plt.figure(figsize=(10, 6))
            sns.scatterplot(data=data, x='rtt_download_sec', y='download_throughput_bps', alpha=0.5)
            plt.title(f'RTT vs Download Throughput\n({name})')
            plt.xlabel('RTT Download (segundos)')
            plt.ylabel('Download Throughput (bps)')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(GRAPHS_FOLDER, f"{name}_rtt_vs_download_scatter.png"))
            plt.close()

        
        if not data['rtt_upload_sec'].empty and not data['upload_throughput_bps'].empty:
            plt.figure(figsize=(10, 6))
            sns.scatterplot(data=data, x='rtt_upload_sec', y='upload_throughput_bps', alpha=0.5)
            plt.title(f'RTT vs Upload Throughput\n({name})')
            plt.xlabel('RTT Upload (segundos)')
            plt.ylabel('Upload Throughput (bps)')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(GRAPHS_FOLDER, f"{name}_rtt_vs_upload_scatter.png"))
            plt.close()

    print(f"Geração de gráficos concluída. Gráficos salvos em: {GRAPHS_FOLDER}")

def main():
    try:
        df = pd.read_csv('ndt_tests_corrigido.csv')
        
        n = 1000
        df['packet_loss_count_n1000'] = (df['packet_loss_percent'] / 100 * n).round().astype(int)
        print("Coluna 'packet_loss_count_n1000' (k para n=1000) criada.")
        

        
        
        calculate_descriptive_stats(df)
        
        
        
        generate_selected_graphs(df)
        
        print(f"\nAnálise Exploratória (EDA) concluída!")
        print(f"Resultados salvos na pasta: {OUTPUT_FOLDER}")
        
    except FileNotFoundError:
        print(f"Erro: O arquivo 'ndt_tests_corrigido.csv' não foi encontrado.")
    except Exception as e:
        print(f"Ocorreu um erro inesperado: {e}")

if __name__ == "__main__":
    main()
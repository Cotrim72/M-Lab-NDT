import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Configurações ---
OUTPUT_FOLDER = "analise_exploratoria"
TABLES_FOLDER = os.path.join(OUTPUT_FOLDER, "tabelas_resumo")
GRAPHS_FOLDER = os.path.join(OUTPUT_FOLDER, "graficos_selecionados")

# Variáveis de interesse
VARIABLES = [
    'download_throughput_bps', 
    'upload_throughput_bps', 
    'rtt_download_sec',
    'rtt_upload_sec', 
    'packet_loss_percent'
]

# --- MODIFICAÇÃO: Adicionar a nova variável de contagem à lista ---
# Isso fará com que as funções de estatística e plotagem a incluam.
VARIABLES.append('packet_loss_count_n1000')
# --- FIM DA MODIFICAÇÃO ---

# Quantis relevantes para análise de rede (cauda da distribuição)
QUANTILES = [0.90, 0.95, 0.99]

# Seleção para análise gráfica (como solicitado)
SELECTED_CLIENT = 'client13'
SELECTED_SERVER = 'server01'


def calculate_descriptive_stats(df):
    """
    Calcula estatísticas descritivas para todas as variáveis,
    agrupadas por cliente e por servidor.
    Salva os resultados em arquivos CSV e imprime um resumo no console.
    """
    print("Iniciando Cálculo de Estatísticas Descritivas...")
    os.makedirs(TABLES_FOLDER, exist_ok=True)
    
    # Lista para armazenar os dataframes de estatísticas
    all_stats = []

    # --- Estatísticas por Cliente ---
    try:
        # Agrupa por cliente e calcula as estatísticas para as variáveis
        client_stats = df.groupby('client')[VARIABLES].agg(
            ['mean', 'median', 'var', 'std']
        )
        
        # Calcula os quantis separadamente
        client_quantiles = df.groupby('client')[VARIABLES].quantile(QUANTILES).unstack()
        
        # Junta as estatísticas e os quantis
        client_summary = pd.concat([client_stats, client_quantiles], axis=1)
        
        # Renomeia colunas para melhor legibilidade
        client_summary.columns = ['_'.join(map(str, col)).strip() for col in client_summary.columns.values]
        
        # Adiciona contagem de amostras
        client_summary['sample_count'] = df.groupby('client').size()
        
        client_summary.to_csv(os.path.join(TABLES_FOLDER, "stats_por_cliente.csv"))
        all_stats.append(("Por Cliente", client_summary))
    except Exception as e:
        print(f"Erro ao calcular estatísticas por cliente: {e}")

    # --- Estatísticas por Servidor ---
    try:
        # Agrupa por servidor e calcula as estatísticas
        server_stats = df.groupby('server')[VARIABLES].agg(
            ['mean', 'median', 'var', 'std']
        )
        
        # Calcula os quantis separadamente
        server_quantiles = df.groupby('server')[VARIABLES].quantile(QUANTILES).unstack()
        
        # Junta as estatísticas e os quantis
        server_summary = pd.concat([server_stats, server_quantiles], axis=1)
        
        # Renomeia colunas
        server_summary.columns = ['_'.join(map(str, col)).strip() for col in server_summary.columns.values]
        
        # Adiciona contagem de amostras
        server_summary['sample_count'] = df.groupby('server').size()
        
        server_summary.to_csv(os.path.join(TABLES_FOLDER, "stats_por_servidor.csv"))
        all_stats.append(("Por Servidor", server_summary))
    except Exception as e:
        print(f"Erro ao calcular estatísticas por servidor: {e}")

    # --- Estatísticas Gerais ---
    try:
        # Calcula estatísticas gerais para o dataset todo
        general_stats = df[VARIABLES].agg(
            ['mean', 'median', 'var', 'std']
        )
        
        # Calcula os quantis gerais
        general_quantiles = df[VARIABLES].quantile(QUANTILES).unstack().T
        general_quantiles.columns = [f'quantile_{q}' for q in QUANTILES]

        # Junta
        general_summary = pd.concat([general_stats, general_quantiles], axis=1)
        
        # Adiciona contagem
        general_summary['sample_count'] = len(df)
        
        general_summary.to_csv(os.path.join(TABLES_FOLDER, "stats_gerais.csv"))
        all_stats.append(("Geral", general_summary))
    except Exception as e:
        print(f"Erro ao calcular estatísticas gerais: {e}")


    print("\n--- Resumo das Estatísticas Geradas ---")
    for name, stats_df in all_stats:
        print(f"\n{name} (primeiras linhas):")
        print(stats_df.head())
        
    print(f"\nTabelas CSV salvas em: {TABLES_FOLDER}")


def generate_selected_graphs(df):
    """
    Gera histogramas, boxplots e scatter plots para o cliente e servidor
    selecionados (client13 e server01).
    Salva os gráficos na pasta de gráficos.
    """
    print("\nIniciando Geração de Gráficos para Entidades Selecionadas...")
    os.makedirs(GRAPHS_FOLDER, exist_ok=True)

    # Configurações de estilo (mantendo o estilo do script original)
    sns.set_theme(style="whitegrid")
    
    # Filtrar dados para as entidades selecionadas
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

        # 1. Histogramas e Boxplots para cada variável
        for var in VARIABLES:
            if data[var].isnull().all():
                print(f"Aviso: Todos os dados são nulos para {var} em {name}. Gráfico pulado.")
                continue

            # --- Histograma ---
            plt.figure(figsize=(10, 6))
            # Para a contagem, queremos barras discretas (histplot faz isso bem)
            # Para as contínuas, kde=True é bom.
            is_discrete = 'count' in var
            
            sns.histplot(data=data, x=var, kde=not is_discrete, discrete=is_discrete, bins=50 if not is_discrete else None)
            
            plt.title(f'Histograma de {var}\n({name})')
            plt.xlabel(f'Valor ({var})')
            plt.ylabel('Frequência')
            plt.tight_layout()
            plt.savefig(os.path.join(GRAPHS_FOLDER, f"{name}_{var}_histogram.png"))
            plt.close()

            # --- Boxplot ---
            plt.figure(figsize=(10, 4))
            sns.boxplot(data=data, x=var, palette="vlag")
            plt.title(f'Boxplot de {var}\n({name})')
            plt.xlabel(f'Valor ({var})')
            plt.tight_layout()
            plt.savefig(os.path.join(GRAPHS_FOLDER, f"{name}_{var}_boxplot.png"))
            plt.close()

        # 2. Scatter Plot: RTT vs Download Throughput
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

        # 3. Scatter Plot: RTT vs Upload Throughput
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
    """
    Função principal para executar a análise exploratória.
    """
    try:
        df = pd.read_csv('ndt_tests_corrigido.csv')
        
        # --- MODIFICAÇÃO: Criar a variável de contagem de perdas (k) ---
        # Assumindo n=1000 e que packet_loss_percent é um valor de 0-100.
        # k = (percent / 100) * 1000 = percent * 10
        n = 1000
        df['packet_loss_count_n1000'] = (df['packet_loss_percent'] / 100 * n).round().astype(int)
        print("Coluna 'packet_loss_count_n1000' (k para n=1000) criada.")
        # --- FIM DA MODIFICAÇÃO ---

        # Parte 1: Calcular Estatísticas Descritivas
        # A nova variável será incluída automaticamente
        calculate_descriptive_stats(df)
        
        # Parte 2: Gerar Gráficos para Entidades Selecionadas
        # A nova variável será plotada automaticamente
        generate_selected_graphs(df)
        
        print(f"\nAnálise Exploratória (EDA) concluída!")
        print(f"Resultados salvos na pasta: {OUTPUT_FOLDER}")
        
    except FileNotFoundError:
        print(f"Erro: O arquivo 'ndt_tests_corrigido.csv' não foi encontrado.")
    except Exception as e:
        print(f"Ocorreu um erro inesperado: {e}")

if __name__ == "__main__":
    main()
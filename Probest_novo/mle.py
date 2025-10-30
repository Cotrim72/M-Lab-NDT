import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import warnings
import os
import seaborn as sns

RESULTS_FOLDER = "mle_etapa_2"
PARAMETER_FILE = os.path.join(RESULTS_FOLDER, "mle_parametros_selecionados.txt")

ENTITIES_TO_ANALYZE = [
    ('client', 'client13'),
    ('server', 'server01')
]
N_BINOMIAL = 1000 

warnings.filterwarnings('ignore')
plt.ioff() 

def analyze_entity(df, entity_type, entity_id, base_folder):
    """
    Executa a análise MLE completa
    """
        
    entity_folder_name = f"{entity_type}_{entity_id.replace(entity_type, '')}"
    entity_folder_path = os.path.join(base_folder, entity_folder_name)
    os.makedirs(entity_folder_path, exist_ok=True)
    
    print(f"\n--- Iniciando Análise MLE para: {entity_type} {entity_id} ---")
    
    results_str = f"\n{'='*30}\n Entidade: {entity_type} {entity_id}\n{'='*30}\n"
        
    entity_df = df[df[entity_type] == entity_id]
    if entity_df.empty:
        print(f"  Aviso: Nenhum dado encontrado para {entity_id}.")
        return f"Nenhum dado para {entity_id}\n"
    
    continuous_models = {
        'download_throughput_bps': {'model': stats.gamma, 'name': 'Download Throughput (Gamma)'},
        'upload_throughput_bps': {'model': stats.gamma, 'name': 'Upload Throughput (Gamma)'},
        'rtt_download_sec': {'model': stats.norm, 'name': 'RTT Download (Normal)'},
        'rtt_upload_sec': {'model': stats.norm, 'name': 'RTT Upload (Normal)'}
    }

    for var, config in continuous_models.items():
        model = config['model']
        name = config['name']
        dist_name_str = model.name 
        
        print(f"  Processando: {name} ({var})")
        
        
        data = entity_df[var].dropna()
        if data.empty:
            print(f"    Aviso: Sem dados para {var}.")
            results_str += f"{name} ({var}): Sem dados\n"
            continue
        
        try:
            
            if model == stats.gamma:
                
                params = model.fit(data, floc=0)
                a, loc, scale = params 
                results_str += f"{name} ({var}):\n"
                results_str += f"  shape (a): {a:.4f}, loc: {loc:.4f}, scale (b): {scale:.4f}\n"
            else: 
                params = model.fit(data)
                mu, std = params
                results_str += f"{name} ({var}):\n"
                results_str += f"  mean (mu): {mu:.4f}, std (sigma): {std:.4f}\n"
            
            plt.figure(figsize=(10, 6))
            sns.histplot(data, bins=50, stat='density', kde=False, label='Dados (Histograma)')
            
            x = np.linspace(data.min(), data.max(), 200)
            pdf = model.pdf(x, *params) 
                
            plt.plot(x, pdf, 'r-', lw=2, label=f'MLE Fit ({model.name})')
            plt.title(f'Histograma e PDF do MLE - {name}\n({entity_id})')
            plt.xlabel('Valor')
            plt.ylabel('Densidade')
            plt.legend()
            plt.tight_layout()
            hist_path = os.path.join(entity_folder_path, f"{var}_hist_pdf.png")
            plt.savefig(hist_path)
            plt.close()
            
            plt.figure(figsize=(8, 6))
            stats.probplot(data, dist=dist_name_str, sparams=params, plot=plt)
                
            plt.title(f'QQ-Plot - {name}\n({entity_id})')
            plt.xlabel(f'Quantis Teóricos ({model.name})')
            plt.ylabel('Quantis dos Dados')
            plt.grid(True)
            plt.tight_layout()
            qq_path = os.path.join(entity_folder_path, f"{var}_qq.png")
            plt.savefig(qq_path)
            plt.close()
            
        except Exception as e:
            
            print(f"    ERRO ao ajustar {var}: {e}")
            results_str += f"{name} ({var}): Erro no ajuste ({e})\n"
            if plt.gcf().get_axes(): 
                plt.close()

    var = 'packet_loss_count_n1000'
    name = f'Packet Loss (Binomial n={N_BINOMIAL})'
    print(f"  Processando: {name} ({var})")
    
    data = entity_df[var].dropna()
    if data.empty:
        print(f"    Aviso: Sem dados para {var}.")
        results_str += f"{name} ({var}): Sem dados\n"
    else:
        
        k_mean = data.mean()
        p_mle = k_mean / N_BINOMIAL
        
        if p_mle == 0: p_mle = 1e-9
        if p_mle == 1: p_mle = 1 - 1e-9
        
        results_str += f"{name} ({var}):\n"
        results_str += f"  n (fixo): {N_BINOMIAL}\n"
        results_str += f"  p_mle: {p_mle:.8f}\n" 
        results_str += f"  (Média obs. k: {k_mean:.4f})\n"
        
        plt.figure(figsize=(10, 6))
        
        if data.max() > data.min():
            bins = np.arange(data.min(), data.max() + 2) - 0.5
            align = 'mid'
        else: 
            bins = 1
            align = 'center'

        plt.hist(data, bins=bins, density=True, label='Dados (Histograma)', align=align, rwidth=0.9)
        
        k_values = np.arange(max(0, data.min()), data.max() + 2)
        pmf = stats.binom.pmf(k_values, N_BINOMIAL, p_mle)
        
        plt.plot(k_values, pmf, 'ro', ms=6, label=f'MLE Fit (Binomial PMF)')
        plt.title(f'Histograma e PMF do MLE - {name}\n({entity_id})')
        plt.xlabel('Contagem de Perdas (k)')
        plt.ylabel('Probabilidade')
        plt.legend()
        plt.xlim(left=max(-1, data.min() - 2), right=data.max() + 2)
        plt.tight_layout()
        hist_path = os.path.join(entity_folder_path, f"{var}_hist_pmf.png")
        plt.savefig(hist_path)
        plt.close()

        plt.figure(figsize=(8, 6))
        try:
            
            stats.probplot(data, dist=stats.binom, sparams=(N_BINOMIAL, p_mle), plot=plt)
            plt.title(f'QQ-Plot - {name}\n({entity_id})')
            plt.xlabel(f'Quantis Teóricos (Binomial)')
            plt.ylabel('Quantis dos Dados (k)')
            plt.grid(True)
        except Exception as e:
            
            plt.text(0.5, 0.5, f'Não foi possível gerar QQ-Plot para {var}:\n{e}', 
                     horizontalalignment='center', verticalalignment='center')
            print(f"    Aviso: Não foi possível gerar QQ-Plot para {var}: {e}")
            
        plt.tight_layout()
        qq_path = os.path.join(entity_folder_path, f"{var}_qq.png")
        plt.savefig(qq_path)
        plt.close()

    print(f"  Análise de {entity_id} concluída.")
    return results_str

if __name__ == "__main__":
    try:
        df = pd.read_csv('ndt_tests_corrigido.csv')
        print("Dados 'ndt_tests_corrigido.csv' carregados com sucesso.")

        
        df['packet_loss_count_n1000'] = (df['packet_loss_percent'] / 100 * N_BINOMIAL).round().astype(int)
        print(f"Coluna 'packet_loss_count_n1000' (k para n={N_BINOMIAL}) criada.")

        
        os.makedirs(RESULTS_FOLDER, exist_ok=True)
        
        
        with open(PARAMETER_FILE, 'w') as f:
            f.write("Resultados da Análise de Máxima Verossimilhança (MLE)\n")
            f.write(f"Modelos: Gamma (Throughput), Normal (RTT), Binomial(n={N_BINOMIAL}) (Packet Loss)\n")
            
            
            for e_type, e_id in ENTITIES_TO_ANALYZE:
                entity_results = analyze_entity(df, e_type, e_id, RESULTS_FOLDER)
                f.write(entity_results) 
        
        print(f"\nAnálise MLE (Etapa 2) concluída.")
        print(f"Arquivo de parâmetros salvo em: '{PARAMETER_FILE}'")
        print(f"Gráficos de diagnóstico salvos na pasta: '{RESULTS_FOLDER}'")

    except FileNotFoundError:
        print(f"Erro: O arquivo 'ndt_tests_corrigido.csv' não foi encontrado.")
    except Exception as e:
        print(f"Ocorreu um erro inesperado no script principal: {e}")
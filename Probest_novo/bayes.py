import pandas as pd
import numpy as np
import scipy.stats as stats
import warnings
import os
from sklearn.model_selection import train_test_split

# --- Configurações Principais ---
RESULTS_FOLDER = "bayesian_etapa_3_4"
PARAMETER_FILE = os.path.join(RESULTS_FOLDER, "bayesian_summary_analysis.txt")
ENTITIES_TO_ANALYZE = [
    ('client', 'client13'),
    ('server', 'server01')
]
N_BINOMIAL = 1000   # N Fixo para o modelo Binomial
TEST_SIZE = 0.3     # Proporção dos dados para teste (30%)
RANDOM_STATE = 42 # Para reprodutibilidade do split

# Suprimir warnings
warnings.filterwarnings('ignore')

# --- Funções de Análise Bayesiana (com formatação corrigida) ---

def bayesian_normal_normal_fixed_var(train_data, test_data, sigma2_mle_fixed):
    mu_0 = 0.0
    sigma2_0 = 1e6
    n = len(train_data)
    if n == 0: return {"erro": "Sem dados de treino"}
    y_bar_train = train_data.mean()
    
    prec_0 = 1 / sigma2_0
    prec_mle = 1 / sigma2_mle_fixed
    prec_n = prec_0 + n * prec_mle
    sigma2_n = 1 / prec_n
    mu_n = sigma2_n * (prec_0 * mu_0 + n * prec_mle * y_bar_train)
    
    mle_mean_mu = y_bar_train
    post_mean_mu = mu_n
    post_var_mu = sigma2_n
    
    pred_mean = mu_n
    pred_var = sigma2_mle_fixed + sigma2_n
    
    test_mean = test_data.mean()
    test_var = test_data.var()
    
    return {
        # --- MODIFICAÇÃO: Formatação de alta precisão ---
        "mle_fixed_params": "sigma2_mle (fixa): {:.8e}".format(sigma2_mle_fixed),
        "mle_parameter_estimate": "mu_mle: {:.10f}".format(mle_mean_mu),
        "params_posterior": "mu_n: {:.10f}, sigma2_n: {:.8e}".format(mu_n, sigma2_n),
        "posterior_mean_param": post_mean_mu,
        "posterior_var_param": post_var_mu,
        "preditiva_mean_data": pred_mean,
        "preditiva_var_data": pred_var,
        "teste_mean": test_mean,
        "teste_var": test_var
    }

def bayesian_gamma_gamma_fixed_k(train_data, test_data, k_mle_fixed):
    a_0 = 0.001
    b_0 = 0.001
    n = len(train_data)
    if n == 0: return {"erro": "Sem dados de treino"}
    y_sum_train = train_data.sum()
    y_bar_train = train_data.mean()

    a_n = a_0 + n * k_mle_fixed
    b_n = b_0 + y_sum_train
    
    # --- MODIFICAÇÃO: Calcular lambda (rate) ---
    mle_rate_lambda = k_mle_fixed / y_bar_train
    post_mean_lambda = a_n / b_n
    post_var_lambda = a_n / (b_n**2)
    
    pred_mean = (k_mle_fixed * b_n) / (a_n - 1) if a_n > 1 else float('nan')
    pred_var = (k_mle_fixed * (k_mle_fixed + a_n - 1) * b_n**2) / ((a_n - 1)**2 * (a_n - 2)) if a_n > 2 else float('nan')
    
    test_mean = test_data.mean()
    test_var = test_data.var()
    
    return {
        # --- MODIFICAÇÃO: Formatação de alta precisão (científica) ---
        "mle_fixed_params": "k_mle (fixo): {:.8f}".format(k_mle_fixed),
        "mle_parameter_estimate": "lambda_mle: {:.8e}".format(mle_rate_lambda), # NOTAÇÃO CIENTÍFICA
        "params_posterior": "a_n: {:.6f}, b_n: {:.6e}".format(a_n, b_n),
        "posterior_mean_param": post_mean_lambda,
        "posterior_var_param": post_var_lambda,
        "preditiva_mean_data": pred_mean,
        "preditiva_var_data": pred_var,
        "teste_mean": test_mean,
        "teste_var": test_var
    }
    
def bayesian_beta_binomial(train_data, test_data, N_fixo):
    a_0 = 1.0
    b_0 = 1.0
    n = len(train_data)
    if n == 0: return {"erro": "Sem dados de treino"}
    
    k_sum_train = train_data.sum()
    k_bar_train = train_data.mean()
    n_total_packets = n * N_fixo
    
    a_n = a_0 + k_sum_train
    b_n = b_0 + (n_total_packets - k_sum_train)
    
    mle_prob_p = k_bar_train / N_fixo
    post_mean_p = a_n / (a_n + b_n)
    post_var_p = (a_n * b_n) / ((a_n + b_n)**2 * (a_n + b_n + 1))
    
    pred_mean = N_fixo * post_mean_p
    pred_var = N_fixo * post_mean_p * (1 - post_mean_p) * ( (a_n + b_n + N_fixo) / (a_n + b_n + 1) )
    
    test_mean = test_data.mean()
    test_var = test_data.var()

    return {
        # --- MODIFICAÇÃO: Formatação de alta precisão ---
        "mle_fixed_params": "N (fixo): {}".format(N_fixo),
        "mle_parameter_estimate": "p_mle: {:.10f}".format(mle_prob_p),
        "params_posterior": "a_n: {:.4f}, b_n: {:.4f}".format(a_n, b_n),
        "posterior_mean_param": post_mean_p,
        "posterior_var_param": post_var_p,
        "preditiva_mean_data": pred_mean,
        "preditiva_var_data": pred_var,
        "teste_mean": test_mean,
        "teste_var": test_var
    }

def calculate_mle_fixed_params(entity_df):
    """
    Calcula os parâmetros MLE (shape da Gamma, var da Normal)
    usando TODOS os dados da entidade (entity_df) para usar como "fixos".
    """
    params = {}
    
    # Gamma (Throughput)
    for var in ['download_throughput_bps', 'upload_throughput_bps']:
        data = entity_df[var].dropna()
        if not data.empty:
            try:
                k, _, _ = stats.gamma.fit(data, floc=0)
                params[var] = {'k_mle_fixed': k}
            except Exception as e:
                print("  Erro no fit Gamma (todos os dados): {}".format(e))
                params[var] = {'k_mle_fixed': 1.0} # Fallback
        else:
            params[var] = {'k_mle_fixed': 1.0} # Fallback
            
    # Normal (RTT)
    for var in ['rtt_download_sec', 'rtt_upload_sec']:
        data = entity_df[var].dropna()
        if not data.empty and data.var() > 0:
            _, std = stats.norm.fit(data)
            params[var] = {'sigma2_mle_fixed': std**2}
        else:
            params[var] = {'sigma2_mle_fixed': 1.0} # Fallback
            
    # Binomial (Não precisa, N é fixo)
    params['packet_loss_count_n1000'] = {'N_fixo': N_BINOMIAL}
            
    return params


# --- Script Principal ---
def main():
    try:
        df = pd.read_csv('ndt_tests_corrigido.csv')
        print("Dados 'ndt_tests_corrigido.csv' carregados.")

        # Criar a variável de contagem (k)
        df['packet_loss_count_n1000'] = (df['packet_loss_percent'] / 100 * N_BINOMIAL).round().astype(int)
        print("Coluna 'packet_loss_count_n1000' (k para n={}) criada.".format(N_BINOMIAL))

        os.makedirs(RESULTS_FOLDER, exist_ok=True)
        
        with open(PARAMETER_FILE, 'w', encoding='utf-8') as f:
            
            f.write("="*60 + "\n")
            f.write("      Análise Bayesiana (Etapas 3 e 4) - Modelo Corrigido\n")
            f.write("="*60 + "\n\n")
            
            f.write("Metodologia (Conforme Roteiro e Solicitação):\n")
            f.write("1. Split dos Dados: 70% Treino, 30% Teste.\n")
            f.write("2. Parâmetros Fixos: Parâmetros de nuisance (shape 'k' da Gamma, variância 'sigma^2' da Normal)\n")
            f.write("   foram calculados via MLE usando *TODOS* os dados da entidade e fixados.\n")
            f.write("3. Modelos (Likelihood + Prior -> Posterior):\n")
            f.write("   - RTT (Normal): Likelihood=Normal(mu, sigma2_mle_fixa), Prior(mu)=Normal(nao_inf) -> Posterior(mu)=Normal\n")
            f.write("   - Throughput (Gamma): Likelihood=Gamma(k_mle_fixo, lambda), Prior(lambda)=Gamma(nao_inf) -> Posterior(lambda)=Gamma\n")
            f.write("   - Packet Loss (Binomial): Likelihood=Binom(N, p), Prior(p)=Beta(uniforme) -> Posterior(p)=Beta\n")

            # --- Dicionário de Modelos ---
            model_map = {
                'download_throughput_bps': {'func': bayesian_gamma_gamma_fixed_k, 'name': 'Throughput (Gamma-Gamma)', 'format_mean': '{:.6f}', 'format_var': '{:.6e}', 'format_param': '{:.8e}'},
                'upload_throughput_bps': {'func': bayesian_gamma_gamma_fixed_k, 'name': 'Throughput (Gamma-Gamma)', 'format_mean': '{:.6f}', 'format_var': '{:.6e}', 'format_param': '{:.8e}'},
                'rtt_download_sec': {'func': bayesian_normal_normal_fixed_var, 'name': 'RTT (Normal-Normal)', 'format_mean': '{:.10f}', 'format_var': '{:.8e}', 'format_param': '{:.10f}'},
                'rtt_upload_sec': {'func': bayesian_normal_normal_fixed_var, 'name': 'RTT (Normal-Normal)', 'format_mean': '{:.10f}', 'format_var': '{:.8e}', 'format_param': '{:.10f}'},
                'packet_loss_count_n1000': {'func': bayesian_beta_binomial, 'name': 'Packet Loss (Beta-Binomial)', 'format_mean': '{:.6f}', 'format_var': '{:.6e}', 'format_param': '{:.10f}'}
            }
            
            # --- Loop por Entidade ---
            for e_type, e_id in ENTITIES_TO_ANALYZE:
                
                print("\n--- Processando Entidade: {} {} ---".format(e_type, e_id))
                f.write("\n\n" + "="*60 + "\n")
                f.write("      Resultados para: {} {}\n".format(e_type.upper(), e_id.upper()))
                f.write("="*60 + "\n")
                
                entity_df = df[df[e_type] == e_id].copy()
                if entity_df.empty:
                    print("  Aviso: Sem dados para {}.".format(e_id))
                    f.write("ERRO: NENHUM DADO ENCONTRADO PARA ESTA ENTIDADE.\n")
                    continue
                    
                # 1. Split dos Dados
                train_df, test_df = train_test_split(entity_df, 
                                                     test_size=TEST_SIZE, 
                                                     random_state=RANDOM_STATE)
                
                f.write("\nTotal de Amostras: {} (Treino: {}, Teste: {})\n".format(len(entity_df), len(train_df), len(test_df)))

                # 2. Calcular Parâmetros MLE Fixos (de *TODOS* os dados da entidade)
                print("  Calculando parâmetros MLE fixos (de todos os dados da entidade)...")
                fixed_mle_params = calculate_mle_fixed_params(entity_df)

                # --- Loop por Variável ---
                for var, model_config in model_map.items():
                    print("  Analisando variável: {}...".format(var))
                    
                    train_data = train_df[var].dropna()
                    test_data = test_df[var].dropna()
                    
                    if train_data.empty:
                        print("    Aviso: Sem dados de treino para {}.".format(var))
                        f.write("\nERRO: Sem dados de treino para {}.\n".format(var))
                        continue
                    
                    args_fixos = fixed_mle_params.get(var, {})

                    # 3. Calcular Posterior e Preditiva
                    results = model_config['func'](train_data, test_data, **args_fixos)
                    
                    if "erro" in results:
                        print("    ERRO: {}".format(results['erro']))
                        f.write("\n--- {} ---\nERRO: {}\n".format(var, results['erro']))
                        continue

                    # 4. Escrever resultados no arquivo (com nomenclatura clara e .format())
                    f.write("\n\n--- Variável: {} (Modelo: {}) ---".format(var, model_config['name']))
                    
                    f.write("\n  Parâmetros Fixos (via MLE de *todos* dados): {}".format(results['mle_fixed_params']))
                    f.write("\n  Parâmetros da Posterior (calculado): {}".format(results['params_posterior']))

                    f.write("\n\n  A. Comparação de Estimativas do PARÂMETRO (Etapa 5):")
                    f.write("\n    Estimativa MLE (do *treino*):    {}".format(results['mle_parameter_estimate']))
                    
                    # --- MODIFICAÇÃO: Usar formatação customizada por variável ---
                    f_param = model_config['format_param']
                    f_var = model_config['format_var']
                    f_mean = model_config['format_mean']

                    f.write(("\n    Média da Posterior (Bayes):    E[param|y] = " + f_param).format(results['posterior_mean_param']))
                    f.write(("\n    Variância da Posterior (Bayes):  Var[param|y] = " + f_var).format(results['posterior_var_param']))
                    
                    f.write("\n\n  B. Comparação Preditiva vs Teste (Etapa 4):")
                    f.write(("\n    Média Preditiva (Bayes):       E[y_novo|y] = " + f_mean).format(results['preditiva_mean_data']))
                    f.write(("\n    Média dos Dados de Teste:      E[y_teste] =  " + f_mean).format(results['teste_mean']))
                    
                    f.write(("\n\n    Variância Preditiva (Bayes):     Var[y_novo|y] = " + f_var).format(results['preditiva_var_data']))
                    f.write(("\n    Variância dos Dados de Teste:    Var[y_teste] =  " + f_var).format(results['teste_var']))

            print("\nAnálise Bayesiana (Etapas 3 e 4) concluída.")
            print("Arquivo de resultados salvo em: '{}'".format(PARAMETER_FILE))

    except FileNotFoundError:
        print("Erro: O arquivo 'ndt_tests_corrigido.csv' não foi encontrado.")
    except Exception as e:
        print("Ocorreu um erro inesperado no script principal: {}".format(e))

if __name__ == "__main__":
    main()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# Configurações gerais de plotagem para os gráficos
plt.rcParams.update({'font.size': 12, 'figure.figsize': (10, 6)})

# 1. Carregar os dados
try:
    df = pd.read_csv('ndt_tests_corrigido.csv')
except FileNotFoundError:
    print("Erro: Arquivo 'ndt_tests_corrigido.csv' não encontrado.")
    # Em um script real, você poderia 'exit()' aqui

print("Dados carregados com sucesso.\n")

# ---
# Modelo 1: Throughput (Gama) para 'download_throughput_bps'
# ---
print("--- Modelo 1: Throughput (Gama) ---")
# Carrega todos os dados da coluna
dados_down_raw = df['download_throughput_bps'].dropna()

# FILTRAGEM: A Dist. Gama SÓ é definida para valores > 0.
# Por isso, criamos um novo set de dados 'dados_down_fit'
# contendo apenas os valores positivos.
dados_down_fit = dados_down_raw[dados_down_raw > 0]
print(f"Usando {len(dados_down_fit)} pontos positivos para o ajuste Gama.")

# 1.1. Estimação MLE (Maximum Likelihood Estimation)
# stats.gamma.fit() encontra os parâmetros que maximizam a verossimilhança.
# floc=0 ("fix location at 0") é uma restrição que colocamos,
# dizendo ao modelo que os dados não podem ser menores que zero.
shape_g, loc_g, scale_g = stats.gamma.fit(dados_down_fit, floc=0)

print(f"\nParâmetros MLE (Gama) para 'download_throughput_bps' (apenas > 0):")
print(f"  Shape (a): {shape_g:.4f}")
print(f"  Scale (b): {scale_g:.4f}")

# 1.2. Gráfico 1: Histograma vs PDF
plt.figure(figsize=(10, 6))
# Plotamos o histograma dos dados (density=True faz a área somar 1)
plt.hist(dados_down_fit, bins=50, density=True, alpha=0.7, label='Dados Reais (Histograma, > 0)')
# Criamos um eixo X para plotar a curva do modelo
x_g = np.linspace(dados_down_fit.min(), dados_down_fit.max(), 1000)
# Calculamos a Função de Densidade (PDF) da Gama com os parâmetros MLE
pdf_g = stats.gamma.pdf(x_g, a=shape_g, loc=loc_g, scale=scale_g)
# Plotamos a curva
plt.plot(x_g, pdf_g, 'r-', lw=2, label='Modelo Gama Ajustado (PDF)')
plt.title('MLE - Gama: download_throughput_bps (Valores > 0)')
plt.xlabel('Download Throughput (bps)')
plt.ylabel('Densidade')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('throughput_gamma_hist.png') # Salva a figura
print("Salvo: throughput_gamma_hist.png")

# 1.3. Gráfico 2: Q-Q Plot (Manual)
plt.figure(figsize=(8, 8))
# Ordenamos nossos dados (quantis observados)
dados_q_g = np.sort(dados_down_fit)
n_g = len(dados_q_g)
# Criamos pontos de probabilidade (ex: 0.01, 0.02, ...)
prob_points_g = (np.arange(n_g) + 0.5) / n_g
# Criamos o objeto da *nossa* distribuição Gama ajustada
dist_gamma = stats.gamma(a=shape_g, loc=loc_g, scale=scale_g)
# Calculamos os quantis teóricos usando a Inversa da CDF (ppf)
teor_q_g = dist_gamma.ppf(prob_points_g)
# Plotamos Teórico (x) vs. Observado (y)
plt.scatter(teor_q_g, dados_q_g, alpha=0.5, label='Dados vs Teórico')
# Adicionamos a linha de referência y=x
min_g, max_g = np.min(teor_q_g), np.max(teor_q_g)
plt.plot([min_g, max_g], [min_g, max_g], 'r-', lw=2, label='Linha y=x')
plt.title('Q-Q Plot: Dados vs. Modelo Gama (Valores > 0)')
plt.xlabel('Quantis Teóricos (Gama MLE)')
plt.ylabel('Quantis dos Dados')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('throughput_gamma_qq.png') # Salva a figura
print("Salvo: throughput_gamma_qq.png\n")


# ---
# Modelo 1.5: Throughput (Gama) para 'upload_throughput_bps'
# ---
print("--- Modelo 1: Throughput (Gama) ---")
# Carrega todos os dados da coluna
dados_up_raw = df['upload_throughput_bps'].dropna()

# FILTRAGEM: A Dist. Gama SÓ é definida para valores > 0.
# Por isso, criamos um novo set de dados 'dados_up_fit'
# contendo apenas os valores positivos.
dados_up_fit = dados_up_raw[dados_up_raw > 0]
print(f"Usando {len(dados_up_fit)} pontos positivos para o ajuste Gama.")

# 1.1. Estimação MLE (Maximum Likelihood Estimation)
# stats.gamma.fit() encontra os parâmetros que maximizam a verossimilhança.
# floc=0 ("fix location at 0") é uma restrição que colocamos,
# dizendo ao modelo que os dados não podem ser menores que zero.
shape_g, loc_g, scale_g = stats.gamma.fit(dados_up_fit, floc=0)

print(f"\nParâmetros MLE (Gama) para 'upload_throughput_bps' (apenas > 0):")
print(f"  Shape (a): {shape_g:.4f}")
print(f"  Scale (b): {scale_g:.4f}")

# 1.2. Gráfico 1: Histograma vs PDF
plt.figure(figsize=(10, 6))
# Plotamos o histograma dos dados (density=True faz a área somar 1)
plt.hist(dados_up_fit, bins=50, density=True, alpha=0.7, label='Dados Reais (Histograma, > 0)')
# Criamos um eixo X para plotar a curva do modelo
x_g = np.linspace(dados_up_fit.min(), dados_up_fit.max(), 1000)
# Calculamos a Função de Densidade (PDF) da Gama com os parâmetros MLE
pdf_g = stats.gamma.pdf(x_g, a=shape_g, loc=loc_g, scale=scale_g)
# Plotamos a curva
plt.plot(x_g, pdf_g, 'r-', lw=2, label='Modelo Gama Ajustado (PDF)')
plt.title('MLE - Gama: upload_throughput_bps (Valores > 0)')
plt.xlabel('Upload Throughput (bps)')
plt.ylabel('Densidade')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('throughput_gamma_hist1.png') # Salva a figura
print("Salvo: throughput_gamma_hist1.png")

# 1.3. Gráfico 2: Q-Q Plot (Manual)
plt.figure(figsize=(8, 8))
# Ordenamos nossos dados (quantis observados)
dados_q_g = np.sort(dados_up_fit)
n_g = len(dados_q_g)
# Criamos pontos de probabilidade (ex: 0.01, 0.02, ...)
prob_points_g = (np.arange(n_g) + 0.5) / n_g
# Criamos o objeto da *nossa* distribuição Gama ajustada
dist_gamma = stats.gamma(a=shape_g, loc=loc_g, scale=scale_g)
# Calculamos os quantis teóricos usando a Inversa da CDF (ppf)
teor_q_g = dist_gamma.ppf(prob_points_g)
# Plotamos Teórico (x) vs. Observado (y)
plt.scatter(teor_q_g, dados_q_g, alpha=0.5, label='Dados vs Teórico')
# Adicionamos a linha de referência y=x
min_g, max_g = np.min(teor_q_g), np.max(teor_q_g)
plt.plot([min_g, max_g], [min_g, max_g], 'r-', lw=2, label='Linha y=x')
plt.title('Q-Q Plot: Dados vs. Modelo Gama (Valores > 0)')
plt.xlabel('Quantis Teóricos (Gama MLE)')
plt.ylabel('Quantis dos Dados')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('throughput_gamma_qq1.png') # Salva a figura
print("Salvo: throughput_gamma_qq1.png\n")


# ---
# Modelo 2: RTT (Normal) para 'rtt_download_sec'
# ---
print("--- Modelo 2: RTT (Normal) ---")
# A Dist. Normal aceita zeros, então não precisamos filtrar.
dados_rtt = df['rtt_download_sec'].dropna()
print(f"Dados de RTT: {len(dados_rtt)} pontos no total.")

# 2.1. Estimação MLE
# Para a Normal, o MLE da média (mu) é a média da amostra
# e o MLE do desvio padrão (std) é o desvio padrão da amostra.
mu_n, std_n = stats.norm.fit(dados_rtt)
print(f"\nParâmetros MLE (Normal) para 'rtt_download_sec':")
print(f"  Média (mu): {mu_n:.4f}")
print(f"  Desvio Padrão (sigma): {std_n:.4f}")

# 2.2. Gráfico 1: Histograma vs PDF
plt.figure(figsize=(10, 6))
plt.hist(dados_rtt, bins=50, density=True, alpha=0.7, label='Dados Reais (Histograma)')
x_n = np.linspace(dados_rtt.min(), dados_rtt.max(), 1000)
pdf_n = stats.norm.pdf(x_n, loc=mu_n, scale=std_n)
plt.plot(x_n, pdf_n, 'r-', lw=2, label='Modelo Normal Ajustado (PDF)')
plt.title('MLE - Normal: rtt_download_sec')
plt.xlabel('RTT Download (segundos)')
plt.ylabel('Densidade')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('rtt_normal_hist.png')
print("Salvo: rtt_normal_hist.png")

# 2.3. Gráfico 2: Q-Q Plot (Manual)
plt.figure(figsize=(8, 8))
dados_q_n = np.sort(dados_rtt)
n_n = len(dados_q_n)
prob_points_n = (np.arange(n_n) + 0.5) / n_n
dist_norm = stats.norm(loc=mu_n, scale=std_n)
teor_q_n = dist_norm.ppf(prob_points_n)
plt.scatter(teor_q_n, dados_q_n, alpha=0.5, label='Dados vs Teórico')
min_n, max_n = np.min(teor_q_n), np.max(teor_q_n)
plt.plot([min_n, max_n], [min_n, max_n], 'r-', lw=2, label='Linha y=x')
plt.title('Q-Q Plot: Dados vs. Modelo Normal')
plt.xlabel('Quantis Teóricos (Normal MLE)')
plt.ylabel('Quantis dos Dados')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('rtt_normal_qq.png')
print("Salvo: rtt_normal_qq.png\n")

# ---
# Modelo 2: RTT (Normal) para 'rtt_upload_sec'
# ---
print("--- Modelo 2: RTT (Normal) ---")
# A Dist. Normal aceita zeros, então não precisamos filtrar.
dados_rtt = df['rtt_upload_sec'].dropna()
print(f"Dados de RTT (upload): {len(dados_rtt)} pontos no total.")

# 2.1. Estimação MLE
# Para a Normal, o MLE da média (mu) é a média da amostra
# e o MLE do desvio padrão (std) é o desvio padrão da amostra.
mu_n, std_n = stats.norm.fit(dados_rtt)
print(f"\nParâmetros MLE (Normal) para 'rtt_upload_sec':")
print(f"  Média (mu): {mu_n:.4f}")
print(f"  Desvio Padrão (sigma): {std_n:.4f}")

# 2.2. Gráfico 1: Histograma vs PDF
plt.figure(figsize=(10, 6))
plt.hist(dados_rtt, bins=50, density=True, alpha=0.7, label='Dados Reais (Histograma)')
x_n = np.linspace(dados_rtt.min(), dados_rtt.max(), 1000)
pdf_n = stats.norm.pdf(x_n, loc=mu_n, scale=std_n)
plt.plot(x_n, pdf_n, 'r-', lw=2, label='Modelo Normal Ajustado (PDF)')
plt.title('MLE - Normal: rtt_upload_sec')
plt.xlabel('RTT Upload (segundos)')
plt.ylabel('Densidade')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('rtt_normal_hist1.png')
print("Salvo: rtt_normal_hist1.png")

# 2.3. Gráfico 2: Q-Q Plot (Manual)
plt.figure(figsize=(8, 8))
dados_q_n = np.sort(dados_rtt)
n_n = len(dados_q_n)
prob_points_n = (np.arange(n_n) + 0.5) / n_n
dist_norm = stats.norm(loc=mu_n, scale=std_n)
teor_q_n = dist_norm.ppf(prob_points_n)
plt.scatter(teor_q_n, dados_q_n, alpha=0.5, label='Dados vs Teórico')
min_n, max_n = np.min(teor_q_n), np.max(teor_q_n)
plt.plot([min_n, max_n], [min_n, max_n], 'r-', lw=2, label='Linha y=x')
plt.title('Q-Q Plot: Dados vs. Modelo Normal')
plt.xlabel('Quantis Teóricos (Normal MLE)')
plt.ylabel('Quantis dos Dados')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('rtt_normal_qq1.png')
print("Salvo: rtt_normal_qq1.png\n")


# ---
# Modelo 3: Perda (Beta) para 'packet_loss_percent'
# ---
print("--- Modelo 3: Perda de Pacotes (Beta) ---")
# 1. Transformar de [0, 100] para [0, 1]
dados_perda_raw = df['packet_loss_percent'].dropna() / 100.0

# 2. FILTRAGEM: A Dist. Beta SÓ é definida para (0, 1) [aberto]
# Por isso, filtramos valores 0.0 (0%) e 1.0 (100%).
dados_perda_fit = dados_perda_raw[(dados_perda_raw > 0) & (dados_perda_raw < 1)] 
print(f"Usando {len(dados_perda_fit)} pontos (0 < perda < 1) para o ajuste Beta.")

# 3.1. Estimação MLE
# floc=0 e fscale=1 fixam o intervalo do modelo em [0, 1].
a_b, b_b, loc_b, scale_b = stats.beta.fit(dados_perda_fit, floc=0, fscale=1)
print(f"\nParâmetros MLE (Beta) para 'packet_loss_percent' (0 < perda < 1):")
print(f"  Shape1 (a): {a_b:.4f}")
print(f"  Shape2 (b): {b_b:.4f}")

# 3.2. Gráfico 1: Histograma vs PDF
plt.figure(figsize=(10, 6))
plt.hist(dados_perda_fit, bins=50, density=True, alpha=0.7, label='Dados Reais (Histograma, 0 < perda < 1)')
x_b = np.linspace(dados_perda_fit.min(), dados_perda_fit.max(), 1000)
pdf_b = stats.beta.pdf(x_b, a=a_b, b=b_b, loc=loc_b, scale=scale_b)
plt.plot(x_b, pdf_b, 'r-', lw=2, label='Modelo Beta Ajustado (PDF)')
plt.title('MLE - Beta: packet_loss_percent (Valores 0 < perda < 1)')
plt.xlabel('Perda de Pacote (proporção)')
plt.ylabel('Densidade')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('perda_beta_hist.png')
print("Salvo: perda_beta_hist.png")

# 3.3. Gráfico 2: Q-Q Plot (Manual)
plt.figure(figsize=(8, 8))
dados_q_b = np.sort(dados_perda_fit)
n_b = len(dados_q_b)
prob_points_b = (np.arange(n_b) + 0.5) / n_b
dist_beta = stats.beta(a=a_b, b=b_b, loc=loc_b, scale=scale_b)
teor_q_b = dist_beta.ppf(prob_points_b)
plt.scatter(teor_q_b, dados_q_b, alpha=0.5, label='Dados vs Teórico')
min_b, max_b = np.min(teor_q_b), np.max(teor_q_b)
plt.plot([min_b, max_b], [min_b, max_b], 'r-', lw=2, label='Linha y=x')
plt.title('Q-Q Plot: Dados vs. Modelo Beta (0 < perda < 1)')
plt.xlabel('Quantis Teóricos (Beta MLE)')
plt.ylabel('Quantis dos Dados')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('perda_beta_qq.png')
print("Salvo: perda_beta_qq.png\n")

print("Processo de MLE concluído. Todos os gráficos foram salvos.")
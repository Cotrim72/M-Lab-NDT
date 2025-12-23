import pandas as pd
import numpy as np
from scipy import stats


def realizar_teste_hipotese():
    # Carregamento e Limpeza de Dados
    try:
        df = pd.read_csv("ndt_tests_corrigido.csv")
    except FileNotFoundError:
        print("Erro: Arquivo 'ndt_tests_corrigido.csv' não encontrado.")
        return

    # Filtragem de valores válidos (estritamente positivos para throughput e RTT)
    df = df[(df["download_throughput_bps"] > 0) & (df["rtt_download_sec"] > 0)]

    # Definição dos clientes para comparação
    c1_id, c2_id = "client13", "server01"

    data_1 = df[df["client"] == c1_id]
    data_2 = df[df["server"] == c2_id]

    n1, n2 = len(data_1), len(data_2)

    # TESTE 1: THROUGHPUT (MODELO GAMA)
    y1 = data_1["download_throughput_bps"].values
    y2 = data_2["download_throughput_bps"].values
    y_pool = np.concatenate([y1, y2])

    y_bar1, y_bar2 = np.mean(y1), np.mean(y2)
    y_bar_pool = np.mean(y_pool)

    # Estimação de k (shape) via MLE para os dados conjuntos (pooled)
    k_mle, _, _ = stats.gamma.fit(y_pool, floc=0)

    # Cálculo do W para Throughput (LRT)
    # W = 2k * [n1 * log(Y_pool/Y1) + n2 * log(Y_pool/Y2)]
    w_tp = (
        2
        * k_mle
        * (n1 * np.log(y_bar_pool / y_bar1) + n2 * np.log(y_bar_pool / y_bar2))
    )
    p_tp = 1 - stats.chi2.cdf(w_tp, df=1)

    # TESTE 2: RTT (MODELO NORMAL)
    r1 = data_1["rtt_download_sec"].values
    r2 = data_2["rtt_download_sec"].values
    r_pool = np.concatenate([r1, r2])

    r_bar1, r_bar2 = np.mean(r1), np.mean(r2)

    # Estimação da variância (sigma^2) via MLE para os dados conjuntos
    _, std_pool = stats.norm.fit(r_pool)
    sigma2_mle = std_pool**2

    # Cálculo do W para RTT (LRT)
    # W = (1/sigma^2) * ( (n1*n2)/(n1+n2) ) * (R1 - R2)^2
    w_rtt = (1 / sigma2_mle) * (n1 * n2 / (n1 + n2)) * (r_bar1 - r_bar2) ** 2
    p_rtt = 1 - stats.chi2.cdf(w_rtt, df=1)

    # Geração do Relatório TXT
    valor_critico = 3.841  # Qui-quadrado com 1 grau de liberdade para alfa = 0.05

    with open("relatorio_teste_hipotese.txt", "w", encoding="utf-8") as f:
        f.write("RELATÓRIO DE TESTE DE HIPÓTESE (PARTE 6)\n")
        f.write("=" * 50 + "\n")
        f.write(f"Comparação: {c1_id} vs {c2_id}\n")
        f.write(f"Amostras: n({c1_id})={n1}, n({c2_id})={n2}\n")
        f.write(f"Valor crítico (W crítico): {valor_critico}\n\n")

        # Seção Throughput
        f.write("6.1 TESTE DE RAZÃO DE VEROSSIMILHANÇA - THROUGHPUT (GAMA)\n")
        f.write("-" * 50 + "\n")
        f.write(f"H0: Beta_{c1_id} = Beta_{c2_id} | H1: Beta_{c1_id} != Beta_{c2_id}\n")
        f.write(f"Média {c1_id}: {y_bar1/1e6:.2f} Mbps\n")
        f.write(f"Média {c2_id}: {y_bar2/1e6:.2f} Mbps\n")
        f.write(f"Parâmetro k (estimado): {k_mle:.4f}\n")
        f.write(f"Estatística W_obs: {w_tp:.4f}\n")
        f.write(f"P-Valor: {p_tp:.4e}\n")
        decisao_tp = "REJEITAR H0" if w_tp > valor_critico else "NÃO REJEITAR H0"
        f.write(f"Decisão (alfa=0.05): {decisao_tp}\n")
        f.write(
            f"Conclusão: {'As velocidades são estatisticamente diferentes.' if w_tp > valor_critico else 'Não há diferença significativa nas velocidades.'}\n\n"
        )

        # Seção RTT
        f.write("6.2 TESTE DE RAZÃO DE VEROSSIMILHANÇA - RTT (NORMAL)\n")
        f.write("-" * 50 + "\n")
        f.write(f"H0: Mu_{c1_id} = Mu_{c2_id} | H1: Mu_{c1_id} != Mu_{c2_id}\n")
        f.write(f"Média RTT {c1_id}: {r_bar1*1000:.2f} ms\n")
        f.write(f"Média RTT {c2_id}: {r_bar2*1000:.2f} ms\n")
        f.write(f"Variância sigma^2 (estimada): {sigma2_mle:.6f}\n")
        f.write(f"Estatística W_obs: {w_rtt:.4f}\n")
        f.write(f"P-Valor: {p_rtt:.4e}\n")
        decisao_rtt = "REJEITAR H0" if w_rtt > valor_critico else "NÃO REJEITAR H0"
        f.write(f"Decisão (alfa=0.05): {decisao_rtt}\n")
        f.write(
            f"Conclusão: {'As latências são estatisticamente diferentes.' if w_rtt > valor_critico else 'Não há diferença significativa nas latências.'}\n"
        )

    print("Cálculos concluídos. Resultados salvos em 'relatorio_teste_hipotese.txt'.")


if __name__ == "__main__":
    realizar_teste_hipotese()

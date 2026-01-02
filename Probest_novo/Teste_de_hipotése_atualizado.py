import pandas as pd
import numpy as np
from scipy import stats


def calculate_lrt_gamma(data1, data2):
    """Calculates LRT for Throughput using Gamma distribution (Shape k fixed via MLE)."""
    n1, n2 = len(data1), len(data2)
    y_pool = np.concatenate([data1, data2])

    y_bar1, y_bar2 = np.mean(data1), np.mean(data2)
    y_bar_pool = np.mean(y_pool)

    # MLE for k (shape) on pooled data
    k_mle, _, _ = stats.gamma.fit(y_pool, floc=0)

    # W = 2k * [n1 * log(Y_pool/Y1) + n2 * log(Y_pool/Y2)]
    w_obs = (
        2
        * k_mle
        * (n1 * np.log(y_bar_pool / y_bar1) + n2 * np.log(y_bar_pool / y_bar2))
    )
    p_value = 1 - stats.chi2.cdf(w_obs, df=1)

    return w_obs, p_value, y_bar1, y_bar2, k_mle


def calculate_lrt_normal(data1, data2):
    """Calculates LRT for RTT using Normal distribution (Testing means with pooled variance)."""
    n1, n2 = len(data1), len(data2)
    r_pool = np.concatenate([data1, data2])

    r_bar1, r_bar2 = np.mean(data1), np.mean(data2)

    # MLE for variance (sigma^2) on pooled data
    _, std_pool = stats.norm.fit(r_pool)
    sigma2_mle = std_pool**2

    # W = (1/sigma^2) * ( (n1*n2)/(n1+n2) ) * (R1 - R2)^2
    w_obs = (1 / sigma2_mle) * (n1 * n2 / (n1 + n2)) * (r_bar1 - r_bar2) ** 2
    p_value = 1 - stats.chi2.cdf(w_obs, df=1)

    return w_obs, p_value, r_bar1, r_bar2, sigma2_mle


def realizar_teste_hipotese():
    try:
        df = pd.read_csv("ndt_tests_corrigido.csv")
    except FileNotFoundError:
        print("Erro: Arquivo 'ndt_tests_corrigido.csv' não encontrado.")
        return

    metrics = [
        "download_throughput_bps",
        "upload_throughput_bps",
        "rtt_download_sec",
        "rtt_upload_sec",
    ]
    for col in metrics:
        df = df[df[col] > 0]

    c1_id, c2_id = "client13", "server01"
    data_1 = df[df["client"] == c1_id]
    data_2 = df[df["server"] == c2_id]
    n1, n2 = len(data_1), len(data_2)

    results = {}

    # Throughput (Gamma)
    results["DL_TP"] = calculate_lrt_gamma(
        data_1["download_throughput_bps"].values,
        data_2["download_throughput_bps"].values,
    )
    results["UL_TP"] = calculate_lrt_gamma(
        data_1["upload_throughput_bps"].values, data_2["upload_throughput_bps"].values
    )

    # RTT (Normal)
    results["DL_RTT"] = calculate_lrt_normal(
        data_1["rtt_download_sec"].values, data_2["rtt_download_sec"].values
    )
    results["UL_RTT"] = calculate_lrt_normal(
        data_1["rtt_upload_sec"].values, data_2["rtt_upload_sec"].values
    )

    # Gera o relatório
    valor_critico = 3.841  # alfa = 0.05, gl = 1

    with open("relatorio_teste_hipotese.txt", "w", encoding="utf-8") as f:
        f.write("RELATÓRIO DE TESTE DE HIPÓTESE COMPLETO (UPLOAD/DOWNLOAD)\n")
        f.write("=" * 60 + "\n")
        f.write(f"Comparação: {c1_id} vs {c2_id}\n")
        f.write(f"Amostras: n1={n1}, n2={n2}\n")
        f.write(f"Valor crítico (Chi-Square): {valor_critico}\n\n")

        # Função auxiliar para formatar a escrita no arquivo
        def report_metric(name, res, unit_scale, unit_name, is_gamma=True):
            w_obs, p_val, m1, m2, param = res
            f.write(f"--- TESTE: {name} ---\n")
            f.write(f"H0: Médias Iguais | H1: Médias Diferentes\n")
            f.write(f"Média {c1_id}: {m1*unit_scale:.2f} {unit_name}\n")
            f.write(f"Média {c2_id}: {m2*unit_scale:.2f} {unit_name}\n")
            f.write(
                f"{'Parâmetro k' if is_gamma else 'Variância sigma^2'}: {param:.6f}\n"
            )
            f.write(f"Estatística W_obs: {w_obs:.4f}\n")
            f.write(f"P-Valor: {p_val:.4e}\n")
            decisao = "REJEITAR H0" if w_obs > valor_critico else "NÃO REJEITAR H0"
            f.write(f"Decisão: {decisao}\n")
            f.write(
                f"Conclusão: {'Diferença significativa' if w_obs > valor_critico else 'Sem diferença significativa'}\n\n"
            )

        # Download Throughput
        report_metric(
            "DOWNLOAD THROUGHPUT (GAMA)", results["DL_TP"], 1e-6, "Mbps", True
        )
        # Upload Throughput
        report_metric("UPLOAD THROUGHPUT (GAMA)", results["UL_TP"], 1e-6, "Mbps", True)
        # Download RTT
        report_metric("DOWNLOAD RTT (NORMAL)", results["DL_RTT"], 1000, "ms", False)
        # Upload RTT
        report_metric("UPLOAD RTT (NORMAL)", results["UL_RTT"], 1000, "ms", False)

    print("Cálculos de DL/UL concluídos. Verifique 'relatorio_teste_hipotese.txt'.")


if __name__ == "__main__":
    realizar_teste_hipotese()

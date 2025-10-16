import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Lendo e limpando o CSV ---
df = pd.read_csv("ndt_tests_corrigido.csv")

# Variáveis de interesse
cols_interesse = [
    "download_throughput_bps",
    "upload_throughput_bps",
    "rtt_download_sec",
    "rtt_upload_sec",
    "packet_loss_percent"
]

# Remover linhas com negativos
df = df[(df[cols_interesse] >= 0).all(axis=1)]

# --- Função para calcular estatísticas ---
def calcular_estatisticas(grupo):
    stats = {}
    for col in cols_interesse:
        serie = grupo[col].dropna()
        if len(serie) == 0:
            continue
        stats[col] = {
            "média": serie.mean(),
            "mediana": serie.median(),
            "variância": serie.var(),
            "desvio_padrão": serie.std(),
            "q90": serie.quantile(0.9),
            "q99": serie.quantile(0.99)
        }
    return stats

# --- Estatísticas gerais ---
estatisticas_geral = calcular_estatisticas(df)

# --- Estatísticas por CLIENTE e por SERVIDOR ---
estatisticas_clientes = {c: calcular_estatisticas(g) for c, g in df.groupby("client")}
estatisticas_servidores = {s: calcular_estatisticas(g) for s, g in df.groupby("server")}

# --- Gerar arquivo TXT ---
with open("estatisticas_descritivas.txt", "w", encoding="utf-8") as f:
    f.write("=== ESTATÍSTICAS DESCRITIVAS GERAIS ===\n\n")
    for var, vals in estatisticas_geral.items():
        f.write(f"{var}:\n")
        for k, v in vals.items():
            f.write(f"  {k:<15}: {v:.4f}\n")
    f.write("\n")

    f.write("=== ESTATÍSTICAS DESCRITIVAS POR CLIENTE ===\n\n")
    for client, vars in estatisticas_clientes.items():
        f.write(f"CLIENTE: {client}\n")
        for var, vals in vars.items():
            f.write(f"  {var}:\n")
            for k, v in vals.items():
                f.write(f"    {k:<15}: {v:.4f}\n")
        f.write("\n")

    f.write("\n=== ESTATÍSTICAS DESCRITIVAS POR SERVIDOR ===\n\n")
    for server, vars in estatisticas_servidores.items():
        f.write(f"SERVIDOR: {server}\n")
        for var, vals in vars.items():
            f.write(f"  {var}:\n")
            for k, v in vals.items():
                f.write(f"    {k:<15}: {v:.4f}\n")
        f.write("\n")

    # --- Texto interpretativo sobre quantis ---
    f.write("\n=== INTERPRETAÇÃO SOBRE OS QUANTIS ===\n")
    f.write("""
Os quantis 0,9 e 0,99 foram calculados para destacar o comportamento da cauda
das distribuições, especialmente relevante em métricas de desempenho de rede
como latência (RTT) e throughput.

• O quantil 0,9 (q90) indica o valor que 90% das observações não ultrapassam,
  representando o 'limite superior' do desempenho típico.

• O quantil 0,99 (q99) mostra o comportamento extremo — os piores 1% dos casos.
  Isso é importante em redes porque mesmo poucos picos de latência ou perdas
  podem degradar a experiência do usuário e revelar gargalos.

Assim, comparar q90 e q99 entre clientes e servidores permite identificar quais
entidades apresentam maior instabilidade ou comportamento de cauda pesada
(não desejável em aplicações sensíveis à latência).
""")

print("Arquivo 'estatisticas_descritivas.txt' gerado com sucesso!")


agrupado = df.groupby(["client", "server"])

estatisticas = agrupado[cols_interesse].agg(
    [
        "mean",     # média
        "median",   # mediana
        "var",      # variância
        "std",      # desvio padrão
        lambda x: x.quantile(0.9),   # quantil 0.9
        lambda x: x.quantile(0.99)   # quantil 0.99
    ]
)

estatisticas.columns = [
    "_".join(col).replace("<lambda_0>", "q90").replace("<lambda_1>", "q99")
    for col in estatisticas.columns
]

pd.options.display.float_format = "{:.3f}".format  # formato bonito
print(estatisticas)

estatisticas.to_csv("Combinações_de_cliente_servidor.csv")
print("\nResumo salvo em 'Combinações_de_cliente_servidor.csv'")

medianas_cliente = df.groupby("client")["download_throughput_bps"].median()
cliente_max = medianas_cliente.idxmax()  # cliente com maior throughput
cliente_min = medianas_cliente.idxmin()  # cliente com menor throughput

selected_clients = [cliente_max, cliente_min]

for client in selected_clients:
    data = df[df["client"] == client]
    prefix = f"Cliente {client}"
    
    # --- Histograma ---
    data[["download_throughput_bps","upload_throughput_bps"]].hist(bins=30, figsize=(10,4))
    plt.suptitle(f"{prefix} — Histograma de Throughput")
    plt.savefig(f"{client}_histogram_throughput.png")
    plt.close()
    
    # --- Boxplot ---
    data[["rtt_download_sec","rtt_upload_sec"]].plot.box(figsize=(6,4))
    plt.title(f"{prefix} — Boxplot de RTT")
    plt.savefig(f"{client}_boxplot_rtt.png")
    plt.close()
    
    # --- Scatter plot ---
    plt.figure(figsize=(6,4))
    plt.scatter(data["download_throughput_bps"], data["rtt_download_sec"], alpha=0.6)
    plt.xlabel("Download Throughput (bps)")
    plt.ylabel("RTT Download (s)")
    plt.title(f"{prefix} — Scatter: Download Throughput vs RTT Download")
    plt.savefig(f"{client}_scatter_download_vs_rtt.png")
    plt.close()


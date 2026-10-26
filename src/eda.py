"""Análise exploratória: qualidade dos dados, 3 descobertas, outliers e checagem de vazamento.

Uso (na raiz do projeto):  python -m src.eda --data data/creditcard.csv
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.common import ARQUIVO_PADRAO, carregar_dados, preparar


def descobertas(df, pasta):
    """Gera os gráficos das 3 descobertas da EDA e devolve os números para o relatório.

     1) desbalanceamento: a fraude é uma fração minúscula, então a acurácia engana;
     2) Amount: muito assimétrico e com comportamento diferente por classe;
     3) as variáveis V que mais diferem entre fraude e normal.
    Também resume a taxa de fraude por hora do dia (descritivo).
    Saída: (dicionário com os números, lista das variáveis V mais diferentes).
    """
    resumo = {}

    # descoberta 1: desbalanceamento
    contagem = df["Class"].value_counts()
    plt.figure(figsize=(5, 4))
    plt.bar(["Normal", "Fraude"], [contagem.get(0, 0), contagem.get(1, 0)], color=["steelblue", "crimson"])
    plt.yscale("log")
    plt.title("Desbalanceamento das classes (escala log)")
    plt.ylabel("Transações")
    plt.tight_layout()
    plt.savefig(pasta / "eda1_classes.png", dpi=120)
    plt.close()
    resumo["descoberta_1_proporcao_fraude"] = float(contagem.get(1, 0) / len(df))

    # descoberta 2: Amount é muito assimétrico e difere entre as classes
    plt.figure(figsize=(6, 4))
    for classe, nome, cor in [(0, "Normal", "steelblue"), (1, "Fraude", "crimson")]:
        plt.hist(np.log1p(df.loc[df["Class"] == classe, "Amount"]), bins=50, alpha=0.55,
                 density=True, label=nome, color=cor)
    plt.title("Amount por classe (log1p, densidade)")
    plt.xlabel("log(1 + Amount)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(pasta / "eda2_amount.png", dpi=120)
    plt.close()
    resumo["descoberta_2_amount"] = {
        "mediana": float(df["Amount"].median()), "media": float(df["Amount"].mean()),
        "maximo": float(df["Amount"].max()),
        "mediana_fraude": float(df.loc[df["Class"] == 1, "Amount"].median()),
        "mediana_normal": float(df.loc[df["Class"] == 0, "Amount"].median()),
    }

    # descoberta 3: quais variáveis V mais diferem entre fraude e normal
    colunas_v = [c for c in df.columns if c.startswith("V")]
    dif = ((df[df["Class"] == 1][colunas_v].mean() - df[df["Class"] == 0][colunas_v].mean()).abs()
           / df[colunas_v].std()).sort_values(ascending=False)
    top = dif.head(5)
    plt.figure(figsize=(6, 4))
    top.sort_values().plot(kind="barh", color="crimson")
    plt.title("Variáveis que mais diferem entre fraude e normal")
    plt.xlabel("|diferença das médias| / desvio-padrão")
    plt.tight_layout()
    plt.savefig(pasta / "eda3_variaveis.png", dpi=120)
    plt.close()
    resumo["descoberta_3_top_variaveis"] = {k: round(float(v), 3) for k, v in top.items()}

    # Time: fraude por hora do dia (só descritivo)
    if "Time" in df.columns:
        hora = ((df["Time"] // 3600) % 24).astype(int)
        por_hora = df.groupby(hora)["Class"].agg(["mean", "size"])
        resumo["fraude_por_hora_do_dia_primeiras"] = {int(h): round(float(m), 5) for h, m in por_hora["mean"].head(24).items()}
    return resumo, list(top.index)


def outliers_iqr(df, colunas, pasta):
    """Conta, por classe, quantas transações são outliers pela regra do IQR.

    Regra: fora de [Q1 - 1,5*IQR, Q3 + 1,5*IQR]. Serve para mostrar que nem todo
    outlier é fraude (compras legítimas de valor alto também aparecem).
    Saída: tabela por variável, também salva em CSV.
    """
    linhas = []
    for coluna in colunas:
        q1, q3 = df[coluna].quantile(0.25), df[coluna].quantile(0.75)
        fora = (df[coluna] < q1 - 1.5 * (q3 - q1)) | (df[coluna] > q3 + 1.5 * (q3 - q1))
        linhas.append({"variavel": coluna,
                       "outliers_normal_%": round(100 * fora[df["Class"] == 0].mean(), 2),
                       "outliers_fraude_%": round(100 * fora[df["Class"] == 1].mean(), 2)})
    tabela = pd.DataFrame(linhas)
    tabela.to_csv(pasta / "eda4_outliers.csv", index=False)
    return tabela


def main():
    """Roda a EDA completa: qualidade dos dados, 3 descobertas, outliers e checagem de vazamento.

    Imprime o resumo no terminal e salva gráficos (PNG) e eda_summary.json em results/.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=ARQUIVO_PADRAO)
    parser.add_argument("--output", default="results")
    args = parser.parse_args()
    pasta = Path(args.output)
    pasta.mkdir(parents=True, exist_ok=True)

    df, info = carregar_dados(args.data)
    print("--- Qualidade dos dados ---")
    for chave, valor in info.items():
        print(f"{chave}: {valor}")

    resumo, top = descobertas(df, pasta)
    print("\n--- Descobertas ---")
    print(f"1) Fraude é {resumo['descoberta_1_proporcao_fraude']:.3%} das transações: a acurácia engana.")
    a = resumo["descoberta_2_amount"]
    print(f"2) Amount: mediana {a['mediana']:.2f}, média {a['media']:.2f}, máximo {a['maximo']:.2f}; "
          f"mediana na fraude {a['mediana_fraude']:.2f} e no normal {a['mediana_normal']:.2f}.")
    print(f"3) Variáveis que mais separam as classes: {resumo['descoberta_3_top_variaveis']}")

    tabela = outliers_iqr(df, ["Amount"] + top[:3], pasta)
    print("\nOutliers (IQR) por classe:\n" + tabela.to_string(index=False))
    print("Leitura: nem todo outlier é fraude; muitas compras legítimas de valor alto aparecem como outlier.")

    dados = preparar(df)
    print(f"\nChecagem de vazamento: linhas idênticas entre treino e teste = {dados.linhas_repetidas}")
    print("Os rótulos não entram no treino; o scaler aprende só com o treino.")

    saida = {"qualidade": info, "eda": resumo, "linhas_repetidas_treino_teste": dados.linhas_repetidas,
             "outliers_iqr": tabela.to_dict(orient="records")}
    (pasta / "eda_summary.json").write_text(json.dumps(saida, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nPronto. Resultados em {pasta}/ (eda_summary.json, eda*.png, eda4_outliers.csv).")


if __name__ == "__main__":
    main()

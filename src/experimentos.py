"""Pipeline ponta a ponta: E1 a E4, ablação de Time, análise de erros e modelo para o protótipo.

Uso (na raiz do projeto):
    python -m src.experimentos --data data/creditcard.csv
Opções:  --sem-lof (pula o LOF, que é lento)   --e4 (roda o experimento opcional das sementes)
         --lof-treino-max 20000 (linhas de treino usadas pelo LOF)
"""
import argparse
import json
import platform
import time
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, precision_recall_curve

from src.common import (ARQUIVO_PADRAO, CONTAMINACAO, SEMENTE, TAMANHO_TESTE, ajustar_zscore, 
                        calcular_metricas, carregar_dados, limiar_orcamento,limiar_por_quantil, 
                        pontuar_if, pontuar_lof, pontuar_zscore, preparar, treinar_if)

# Hipóteses registradas ANTES de rodar (mesmas do README e da ficha de experimentos).
HIPOTESES = {
    "E1": "H1: o Isolation Forest tem PR-AUC maior que o Z-score e o LOF, sem usar rótulos no treino.",
    "E2": "H2: o tempo de treino do Isolation Forest depende de n_estimators e max_samples, e não do "
          "número de registros; o custo do LOF está na predição; com max_samples pequeno a PR-AUC se mantém.",
    "E3": "H3: contamination só move o limiar (o ranking dos scores não muda); aumentá-lo sobe o recall e "
          "derruba a precisão; a acurácia continua alta mesmo com recall baixo.",
    "E4": "Opcional: o resultado do Isolation Forest varia pouco entre sementes aleatórias diferentes.",
    "Extra": "Time (segundos desde a primeira transação) não é comportamento do cartão; "
             "retirá-lo não deve piorar a PR-AUC de forma relevante.",
}


def salvar_grafico(nome, pasta):
    """Ajusta o layout, salva a figura atual como PNG em `pasta` e a fecha."""
    plt.tight_layout()
    plt.savefig(pasta / nome, dpi=120)
    plt.close()


# ===================== E1: comparação dos detectores =====================

def experimento_1(d, usar_lof, lof_max, pasta):
    """E1 (hipótese H1): compara Isolation Forest, Z-score e LOF no mesmo teste.

    Cada detector gera um score por transação. A PR-AUC (independente de limiar) é a
    métrica principal; precisão/recall/F1 usam o MESMO orçamento de alertas para
    todos. Também mede tempo de treino e de predição.
    Linhas de referência: Z-score com |z| >= 3 fixo (mostra quantos alertas gera sem
    orçamento) e 'Sempre normal' (acurácia alta, mas não detecta nada).
    Saída: (modelo IF treinado, scores IF do teste, scores LOF do teste ou None).
    """
    print("\n=== E1: Isolation Forest x Z-score" + (" x LOF" if usar_lof else "") + " ===")
    print(HIPOTESES["E1"])
    y = d.y_teste.to_numpy()

    # Z-score
    t0 = time.perf_counter()
    medias, desvios = ajustar_zscore(d.X_treino)
    t_fit_z = time.perf_counter() - t0
    t0 = time.perf_counter()
    z_teste = pontuar_zscore(d.X_teste, medias, desvios)
    t_pred_z = time.perf_counter() - t0

    # Isolation Forest
    modelo, t_fit_if = treinar_if(d.X_treino)
    t0 = time.perf_counter()
    if_teste = pontuar_if(modelo, d.X_teste)
    t_pred_if = time.perf_counter() - t0

    detectores = {"Isolation Forest": (if_teste, t_fit_if, t_pred_if),
                  "Z-score": (z_teste, t_fit_z, t_pred_z)}
    lof_teste = None
    if usar_lof:
        lof_teste, t_fit_l, t_pred_l = pontuar_lof(d.X_treino, d.X_teste, lof_max)
        detectores["LOF"] = (lof_teste, t_fit_l, t_pred_l)

    linhas = []
    plt.figure(figsize=(6, 4.5))
    # para cada detector: métricas com o mesmo orçamento de alertas + curva precisão x recall
    for nome, (scores, t_fit, t_pred) in detectores.items():
        # mesmo orçamento de alertas para todos: comparação justa de precisão/recall/F1
        m = calcular_metricas(y, scores, limiar_orcamento(scores, CONTAMINACAO))
        linhas.append({"metodo": nome, **m, "tempo_treino_s": t_fit, "tempo_predicao_s": t_pred})
        precisao, recall, _ = precision_recall_curve(y, scores)
        plt.plot(recall, precisao, label=f"{nome} (PR-AUC {m['pr_auc']:.3f})")

    # referência: regra fixa |z| >= 3, para mostrar quantos alertas ela gera sem orçamento
    m = calcular_metricas(y, z_teste, 3.0)
    linhas.append({"metodo": "Z-score (|z|>=3 fixo)", **m, "tempo_treino_s": t_fit_z, "tempo_predicao_s": t_pred_z})
    # referência: modelo que nunca aponta fraude
    sempre = np.zeros(len(y))
    linhas.append({"metodo": "Sempre normal", "pr_auc": float(y.mean()), "roc_auc": 0.5, "precisao": 0.0,
                   "recall": 0.0, "f1": 0.0, "acuracia": float((y == sempre).mean()), "n_alertas": 0,
                   "tempo_treino_s": 0.0, "tempo_predicao_s": 0.0})

    plt.xlabel("Recall"); plt.ylabel("Precisão"); plt.title("E1: curva precisão x recall"); plt.legend()
    salvar_grafico("e1_curva_pr.png", pasta)
    tabela = pd.DataFrame(linhas).set_index("metodo").round(4)
    tabela.to_csv(pasta / "e1_resultados.csv")
    print(tabela.to_string())
    print(f"(P/R/F1 usam o mesmo orçamento de alertas: {CONTAMINACAO:.2%} do teste.)")
    return modelo, if_teste, lof_teste


# ===================== E2: parâmetros e tamanho do treino =====================

def experimento_2(d, pasta):
    """E2 (hipótese H2): muda UM fator por vez no Isolation Forest.

    Fatores: n_estimators, max_samples e o número de registros usados no treino.
    Para cada valor registra PR-AUC, tempo de treino e tempo de predição. Com
    max_samples fixo, o tempo de treino não deve crescer com o número de registros.
    """
    print("\n=== E2: n_estimators, max_samples e tamanho do treino (um fator por vez) ===")
    print(HIPOTESES["E2"])
    y = d.y_teste.to_numpy()
    linhas = []

    def rodar(fator, valor, X_tr, **params):
        """Treina um Isolation Forest com os parâmetros dados e registra PR-AUC e tempos."""
        modelo, t_fit = treinar_if(X_tr, **params)
        t0 = time.perf_counter()
        scores = pontuar_if(modelo, d.X_teste)
        linhas.append({"fator": fator, "valor": valor, "tempo_treino_s": t_fit,
                       "tempo_predicao_s": time.perf_counter() - t0,
                       "pr_auc": average_precision_score(y, scores)})

    for n in [25, 50, 100, 200, 400]:
        rodar("n_estimators", n, d.X_treino, n_estimators=n)
    for ms in [64, 128, 256, 512, 1024, 2048]:
        rodar("max_samples", ms, d.X_treino, max_samples=ms)
    # tamanho do treino com max_samples fixo: o tempo de treino não deve crescer com os registros
    for frac in [0.1, 0.25, 0.5, 1.0]:
        amostra = d.X_treino.sample(frac=frac, random_state=SEMENTE) if frac < 1 else d.X_treino
        rodar("registros_de_treino", len(amostra), amostra, max_samples=256)

    tabela = pd.DataFrame(linhas).round(4)
    tabela.to_csv(pasta / "e2_resultados.csv", index=False)
    print(tabela.to_string(index=False))

    fig, eixos = plt.subplots(1, 3, figsize=(14, 4))
    for eixo, fator in zip(eixos, ["n_estimators", "max_samples", "registros_de_treino"]):
        parte = tabela[tabela["fator"] == fator]
        eixo.plot(parte["valor"], parte["pr_auc"], "o-", color="crimson") 
        eixo.set_xlabel(fator); eixo.set_ylabel("PR-AUC", color="crimson")
        eixo2 = eixo.twinx()
        eixo2.plot(parte["valor"], parte["tempo_treino_s"], "s--", color="steelblue")
        eixo2.set_ylabel("Tempo de treino (s)", color="steelblue")
        eixo.set_title(f"E2: {fator}")
        eixo.set_ylim(0, 0.25)                                  # mesma escala da PR-AUC nos 3 painéis
        eixo2.set_ylim(0, 1.1 * tabela["tempo_treino_s"].max())  # tempo começa em zero
        eixo.tick_params(axis="x", labelrotation=30)
    salvar_grafico("e2_grafico.png", pasta)


# ===================== E3: limiar / contamination =====================

def experimento_3(d, modelo, if_teste, pasta):
    """E3 (hipótese H3): efeito do limiar (`contamination`) sem retreinar o modelo.

    Usa os MESMOS scores e só troca o limiar (quantil dos scores de treino). A PR-AUC
    sai igual em todas as linhas, porque o ranking não muda. Mostra o trade-off
    precisão x recall e que a acurácia continua alta mesmo com pouco recall.
    """
    print("\n=== E3: limiar / contamination ===")
    print(HIPOTESES["E3"])
    y = d.y_teste.to_numpy()
    if_treino = pontuar_if(modelo, d.X_treino)   # o limiar sai do TREINO, não do teste
    linhas = []
    for cont in [0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05]:
        m = calcular_metricas(y, if_teste, limiar_por_quantil(if_treino, cont))
        linhas.append({"contamination": cont, **{k: m[k] for k in
                       ["n_alertas", "precisao", "recall", "f1", "acuracia", "pr_auc"]}})
    tabela = pd.DataFrame(linhas).round(4)
    tabela.to_csv(pasta / "e3_resultados.csv", index=False)
    print(tabela.to_string(index=False))
    print("A coluna pr_auc é igual em todas as linhas: o ranking não muda, só o limiar.")
    print(f"Referência: um modelo que nunca aponta fraude tem acurácia {1 - y.mean():.4f}.")

    plt.figure(figsize=(6.5, 4.5))
    for coluna in ["precisao", "recall", "f1", "acuracia"]:
        plt.plot(tabela["contamination"], tabela[coluna], "o-", label=coluna)
    plt.xscale("log"); plt.xlabel("contamination"); plt.title("E3: efeito do limiar"); plt.legend()
    salvar_grafico("e3_grafico.png", pasta)


# ===================== E4 (opcional) e ablação de Time =====================

def experimento_4(d, pasta):
    """E4 (opcional): repete o Isolation Forest com 10 sementes e mede a variação da PR-AUC.

    Como o algoritmo é aleatório, isso mostra o quanto o resultado depende da sorte.
    """
    print("\n=== E4 (opcional): estabilidade com 10 sementes ===")
    print(HIPOTESES["E4"])
    y = d.y_teste.to_numpy()
    valores = []
    for semente in range(10):
        modelo, _ = treinar_if(d.X_treino, semente=semente)
        valores.append(average_precision_score(y, pontuar_if(modelo, d.X_teste)))
    tabela = pd.DataFrame({"semente": range(10), "pr_auc": np.round(valores, 4)})
    tabela.to_csv(pasta / "e4_resultados.csv", index=False)
    print(tabela.to_string(index=False))
    print(f"PR-AUC média {np.mean(valores):.4f} | desvio-padrão {np.std(valores):.4f}")


def ablacao_time(df, d, pr_auc_com_time, pasta):
    """Extra: retreina o Isolation Forest SEM a variável Time e compara a PR-AUC.

    Time é o tempo desde a primeira transação, não um comportamento do cartão; a
    ablação mostra se o modelo estava usando essa variável. O split é o mesmo.
    """
    print("\n=== Extra: Isolation Forest sem a variável Time ===")
    print(HIPOTESES["Extra"])
    d2 = preparar(df, sem_time=True)
    assert d2.y_teste.index.equals(d.y_teste.index), "o split deveria ser o mesmo"
    modelo, _ = treinar_if(d2.X_treino)
    pr = float(average_precision_score(d2.y_teste, pontuar_if(modelo, d2.X_teste)))
    tabela = pd.DataFrame([{"variaveis": "com Time", "pr_auc": round(pr_auc_com_time, 4)},
                           {"variaveis": "sem Time", "pr_auc": round(pr, 4)}])
    tabela.to_csv(pasta / "extra_sem_time.csv", index=False)
    print(tabela.to_string(index=False))

# ===================== análise de erros =====================

def analisar_erros(d, df, if_teste, lof_teste, pasta):
    """Seleciona casos para a análise de erros (mínimo de 3 exigido pelo checkpoint).

    Casos: 3 falsos positivos (normais com maior score), 3 falsos negativos (fraudes
    com menor score) e, se o LOF rodou, fraudes que o LOF ranqueia melhor que o IF.
    Para cada caso salva posição no ranking, score, Time, Amount e as variáveis mais
    extremas. A coluna `interpretacao_do_grupo` é para o grupo explicar o motivo.
    """
    print("\n=== Análise de erros (3 falsos positivos, 3 falsos negativos e casos de desacordo) ===")
    limiar = limiar_orcamento(if_teste, CONTAMINACAO)
    # tabela com score, rótulo real, posição no ranking (1 = mais suspeita) e decisão do modelo
    t = pd.DataFrame({"score": if_teste, "real": d.y_teste.values}, index=d.X_teste.index)
    t["posicao"] = t["score"].rank(ascending=False, method="min").astype(int)
    t["previsto"] = (t["score"] >= limiar).astype(int)

    casos = [("falso positivo", t[(t.real == 0) & (t.previsto == 1)].sort_values("score", ascending=False).head(3)),
             ("falso negativo", t[(t.real == 1) & (t.previsto == 0)].sort_values("score").head(3))]
    if lof_teste is not None:
        t["posicao_lof"] = pd.Series(lof_teste, index=t.index).rank(ascending=False, method="min").astype(int)
        t["ganho_lof"] = t["posicao"] - t["posicao_lof"]
        casos.append(("fraude que o LOF ranqueia melhor que o IF",
                      t[t.real == 1].sort_values("ganho_lof", ascending=False).head(3)))

    linhas = []
    for tipo, parte in casos:
        for indice, linha in parte.iterrows():
            valores = d.X_teste.loc[indice].drop(["Time", "Amount"], errors="ignore")
            extremas = valores.reindex(valores.abs().sort_values(ascending=False).index).head(3)
            linhas.append({
                "tipo": tipo, "indice": indice, "posicao_no_ranking_IF": int(linha["posicao"],),
                "posicao_no_ranking_LOF": int(t.loc[indice, "posicao_lof"]) if "posicao_lof" in t.columns else "",
                "score_IF": round(float(linha["score"]), 4), "limiar": round(limiar, 4),
                "Time": df.loc[indice, "Time"], "Amount": df.loc[indice, "Amount"],
                "variaveis_mais_extremas": ", ".join(f"{k}={v:.2f}" for k, v in extremas.items()),
                "interpretacao_do_grupo": "[PREENCHER: por que o modelo errou/acertou aqui?]"}),
    erros = pd.DataFrame(linhas)
    erros.to_csv(pasta / "analise_de_erros.csv", index=False)
    print(erros.drop(columns=["interpretacao_do_grupo"]).to_string(index=False) if len(erros)
          else "Nenhum erro desse tipo com esse limiar.")
    print("Para o relatório: escreva a interpretação de cada caso (coluna interpretacao_do_grupo).")


# ===================== modelo para o protótipo =====================

def salvar_modelo_para_app(d, df, modelo, if_teste, pasta_modelo):
    """Guarda em modelo/modelo_impostor.joblib tudo o que o protótipo (app.py) precisa.

    Inclui o modelo, o scaler, o limiar operacional, o PCA para o gráfico 2D (e a
    variância que ele explica), os scores de referência do teste e o 'lote de
    demonstração' (400 normais + até 20 fraudes) usado no ranking de risco.
    """
    if_treino = pontuar_if(modelo, d.X_treino)
    limiar = limiar_por_quantil(if_treino, CONTAMINACAO)

    # PCA só para desenhar o mapa 2D no app (o modelo NÃO usa o PCA para decidir)
    pca = PCA(n_components=2, random_state=SEMENTE).fit(d.X_treino)
    amostra = d.X_teste.sample(min(3000, len(d.X_teste)), random_state=SEMENTE)
    pontos = pca.transform(amostra)
    grafico = pd.DataFrame({"x": pontos[:, 0], "y": pontos[:, 1],
                            "impostor": pontuar_if(modelo, amostra) >= limiar})

    # lote de demonstração: 400 normais + até 20 fraudes do teste (fraudes sobre-representadas de propósito)
    rng = np.random.RandomState(SEMENTE)
    normais = d.y_teste[d.y_teste == 0].index.to_numpy()
    fraudes = d.y_teste[d.y_teste == 1].index.to_numpy()
    escolhidos = np.concatenate([rng.choice(normais, min(400, len(normais)), replace=False),
                                 rng.choice(fraudes, len(fraudes), replace=False)])
    lote = df.loc[escolhidos].copy()
    lote["score"] = pontuar_if(modelo, d.X_teste.loc[escolhidos])
    lote = lote.sort_values("score", ascending=False)
    
    # ranking REAL: as 300 transações mais suspeitas do teste inteiro (escolhidas só pelo score, sem olhar o rótulo)
    scores_todos = pd.Series(if_teste, index=d.X_teste.index)
    top_idx = scores_todos.sort_values(ascending=False).head(300).index
    ranking_real = df.loc[top_idx].copy()
    ranking_real["score"] = scores_todos.loc[top_idx].to_numpy()

    pacote = {"modelo": modelo, "scaler": d.scaler, "colunas": d.colunas, "pca": pca,
              "variancia_pca": [float(v) for v in pca.explained_variance_ratio_],
              "limiar": limiar, "grafico": grafico, "lote": lote,
              "scores_referencia": if_teste, "media_treino": d.X_treino.mean(),
              "desvio_treino": d.X_treino.std().replace(0, 1)}
    pasta_modelo.mkdir(parents=True, exist_ok=True)
    joblib.dump(pacote, pasta_modelo / "modelo_impostor.joblib")
    print(f"\nModelo salvo em {pasta_modelo}/modelo_impostor.joblib (o app.py usa esse arquivo).")


def main():
    """Orquestra o pipeline: carrega, prepara, roda E1 a E3 (E4 se pedido), ablação, erros e salva o modelo.

    Também grava results/run_info.json com versões, parâmetros e contagens.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=ARQUIVO_PADRAO)
    parser.add_argument("--output", default="results")
    parser.add_argument("--model-dir", default="modelo")
    parser.add_argument("--sem-lof", action="store_true", help="não roda o LOF (lento)")
    parser.add_argument("--e4", action="store_true", help="roda o E4 opcional (10 sementes)")
    parser.add_argument("--lof-treino-max", type=int, default=20000)
    args = parser.parse_args()
    pasta = Path(args.output)
    pasta.mkdir(parents=True, exist_ok=True)

    # 1) carregar e limpar (duplicatas removidas e contadas)
    df, info = carregar_dados(args.data)
    print(f"Transações: {info['linhas_originais']} | duplicadas removidas: {info['linhas_duplicadas']} | "
          f"ficaram {info['linhas_apos_remover_duplicadas']} com {info['fraudes_apos_remover_duplicadas']} fraudes")
    # 2) split 70/30 estratificado + padronização aprendida só no treino
    d = preparar(df)
    print(f"Treino: {len(d.X_treino)} | Teste: {len(d.X_teste)} | Fraudes no teste: {int(d.y_teste.sum())} | "
          f"linhas idênticas treino/teste: {d.linhas_repetidas}")

    # 3) experimentos: E1 compara os métodos, E2 mexe nos parâmetros, E3 mexe no limiar
    modelo, if_teste, lof_teste = experimento_1(d, not args.sem_lof, args.lof_treino_max, pasta)
    experimento_2(d, pasta)
    experimento_3(d, modelo, if_teste, pasta)
    if args.e4:
        experimento_4(d, pasta)
    # 4) extras: ablação de Time, análise de erros e modelo para o protótipo
    ablacao_time(df, d, float(average_precision_score(d.y_teste, if_teste)), pasta)
    analisar_erros(d, df, if_teste, lof_teste, pasta)
    salvar_modelo_para_app(d, df, modelo, if_teste, Path(args.model_dir))

    run_info = {"data_hora": datetime.now().isoformat(timespec="seconds"), "python": platform.python_version(),
                "pandas": pd.__version__, "numpy": np.__version__, "scikit_learn": sklearn.__version__,
                "semente": SEMENTE, "tamanho_teste": TAMANHO_TESTE, "contaminacao": CONTAMINACAO,
                "lof_rodou": not args.sem_lof, "lof_treino_max": args.lof_treino_max,
                "e4_rodou": args.e4, "dados": info, "linhas_repetidas_treino_teste": d.linhas_repetidas}
    (pasta / "run_info.json").write_text(json.dumps(run_info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nPronto! Tabelas e gráficos em {pasta}/. Versões e parâmetros em run_info.json.")


if __name__ == "__main__":
    main()
# Protótipo "Central de Impostores" - Isolation Forest
# Antes de abrir, gere o modelo:  python -m src.experimentos --data data/creditcard.csv
# Para abrir:                      python -m streamlit run app.py

import json
import math
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Central de Impostores", layout="wide")

# ---------- identidade visual (um bloco só; o tema base está em .streamlit/config.toml) ----------
TINTA, GRAFITE, REGUA = "#14213D", "#5B6678", "#D5DBE3"
AZUL, ALERTA, ATENCAO, NORMAL = "#1F3A5F", "#B3261E", "#9A6700", "#1B7F4B"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "axes.edgecolor": REGUA,
    "axes.labelcolor": GRAFITE, "axes.titleweight": "600", "axes.titlesize": 11, "axes.labelsize": 9,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": "#E6EAF0", "grid.linewidth": 0.8, "xtick.color": GRAFITE, "ytick.color": GRAFITE,
    "text.color": TINTA, "legend.frameon": False, "font.family": "sans-serif",
})

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&display=swap');
.stApp, .stApp p, .stApp li, .stApp label, .stApp button, .stApp input, .stApp h1, .stApp h2, .stApp h3,
.stApp td, .stApp th, .veredito, [data-testid="stMetricValue"], [data-testid="stMetricLabel"] {
    font-family: 'IBM Plex Sans', system-ui, sans-serif;
}
.block-container, [data-testid="stMainBlockContainer"] { max-width: 1180px; padding-top: 3rem; }
h1 { font-weight: 600; letter-spacing: -0.01em; font-size: 2.1rem; }
h2, h3 { font-weight: 600; letter-spacing: -0.005em; }
[data-testid="stMetric"] { border-top: 2px solid #14213D; padding-top: .6rem; }
[data-testid="stMetricLabel"] { color: #5B6678; }
[data-testid="stMetricValue"] { font-weight: 600; font-variant-numeric: tabular-nums; }
.stButton > button { border-radius: 3px; font-weight: 500; }
[data-testid="stExpander"] { border-radius: 3px; }
.veredito { border-left: 4px solid var(--cor); background: #fff; padding: .85rem 1.1rem;
            margin: .75rem 0 1.25rem; border-radius: 2px; color: #14213D; }
.veredito b { color: var(--cor); font-weight: 600; }
.veredito.alerta { --cor: #B3261E; } .veredito.atencao { --cor: #9A6700; } .veredito.normal { --cor: #1B7F4B; }
</style>
""", unsafe_allow_html=True)


def veredito(tipo, titulo, texto):
    """Faixa de resultado com a cor do estado (alerta, atencao ou normal)."""
    st.markdown(f'<div class="veredito {tipo}"><b>{titulo}</b> {texto}</div>', unsafe_allow_html=True)


@st.cache_resource
def carregar_pacote():
    """Carrega (uma única vez, via cache) o arquivo gerado por src.experimentos.

    Se o arquivo não existir, mostra a instrução de como gerá-lo e para o app.
    """
    caminho = Path("modelo/modelo_impostor.joblib")
    if not caminho.exists():
        st.error("Não achei modelo/modelo_impostor.joblib. Rode primeiro: python -m src.experimentos --data data/creditcard.csv")
        st.stop()
    return joblib.load(caminho)


def analisar(linha, amount, pacote):
    """Pontua UMA transação (valores originais) com o Amount que está na tela.

    Passos: troca o Amount pelo valor digitado; padroniza com o mesmo scaler do treino;
    calcula o score do Isolation Forest e compara com o limiar; projeta o ponto no
    PCA 2D; calcula em que percentual do teste ela é mais suspeita; e lista as 5
    variáveis mais fora do comum (em desvios-padrão).
    """
    colunas = pacote["colunas"]
    dados = linha[colunas].astype(float).copy()
    dados["Amount"] = amount
    transacao = pd.DataFrame([dados.values], columns=colunas)
    padronizada = pd.DataFrame(pacote["scaler"].transform(transacao), columns=colunas)

    # score: sinal invertido para que maior = mais suspeito
    score = float(-pacote["modelo"].score_samples(padronizada)[0])
    # decisão: impostor se o score passa do limiar calculado no treino
    impostor = score >= pacote["limiar"]
    ponto = pacote["pca"].transform(padronizada)[0]
    # percentual do conjunto de teste que é MENOS suspeito que esta transação
    percentil = float((pacote["scores_referencia"] < score).mean())
    z = ((padronizada.iloc[0] - pacote["media_treino"]) / pacote["desvio_treino"]).abs()
    return score, impostor, ponto, percentil, z.sort_values(ascending=False).head(5)


def desenhar_grafico(pacote, ponto, impostor):
    """Desenha o mapa 2D (PCA): tripulantes em cinza, impostores do modelo em salmão e a transação analisada como estrela (vermelha se impostor, verde se normal)."""
    grafico = pacote["grafico"]
    fig, eixo = plt.subplots(figsize=(6, 4.5))
    normais, suspeitos = grafico[~grafico["impostor"]], grafico[grafico["impostor"]]
    eixo.scatter(normais["x"], normais["y"], s=6, c="#C9D0DA", label="Tripulantes")
    eixo.scatter(suspeitos["x"], suspeitos["y"], s=14, c=ALERTA, alpha=0.55, label="Impostores (modelo)")
    eixo.scatter([ponto[0]], [ponto[1]], s=220, c=ALERTA if impostor else NORMAL,
                 edgecolors=TINTA, marker="*", label="Transação analisada")
    var = pacote["variancia_pca"]
    eixo.set_xlabel(f"Componente 1 ({var[0]:.0%} da variância)")
    eixo.set_ylabel(f"Componente 2 ({var[1]:.0%} da variância)")
    eixo.legend(loc="best", fontsize=8)
    fig.tight_layout()
    return fig


def mostrar_resultado(linha, amount, pacote):
    """Analisa a transação e mostra tudo na tela: veredito, métricas, gráfico, variáveis extremas e gabarito.

    O gabarito só aparece se o Amount não foi editado, pois a classe real do dataset
    vale para a transação original.
    """
    score, impostor, ponto, percentil, extremas = analisar(linha, amount, pacote)
    if impostor:
        veredito("alerta", "Impostor.", "Essa transação foge do padrão. É um alerta para revisão, não confirma fraude.")
    elif percentil >= 0.99:
        veredito("atencao", "Tripulante, em zona de atenção.", "O score ficou logo abaixo do limiar.")
    else:
        veredito("normal", "Tripulante.", "Essa transação parece normal.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Score de anomalia", f"{score:.3f}")
    c2.metric("Limiar do modelo", f"{pacote['limiar']:.3f}")
    c3.metric("Mais suspeita que", f"{percentil:.1%} do teste")
    st.pyplot(desenhar_grafico(pacote, ponto, impostor))
    st.caption(f"Esta transação está entre as {1 - percentil:.1%} mais suspeitas do teste; "
               f"o limiar marca como impostor só as {0.002:.1%} mais suspeitas.")
    st.write("Variáveis mais fora do comum nessa transação (em desvios-padrão):")
    st.dataframe(extremas.round(2).to_frame("desvios"))
    with st.expander("Gabarito (classe real do dataset)"):
        if abs(amount - float(linha["Amount"])) > 1e-9:
            st.write("O Amount foi editado, então não existe gabarito para essa transação.")
        else:
            st.write("Fraude de verdade" if int(linha["Class"]) == 1 else "Transação normal")


def aba_analisar(pacote):
    """Aba 'Analisar transação': SORTEAR escolhe uma transação do lote; o usuário pode editar o Amount; ANALISAR mostra o resultado.

    Usa session_state para lembrar a transação sorteada entre as interações da página.
    """
    lote = pacote["lote"]
    if "linha" not in st.session_state:
        st.session_state["linha"], st.session_state["rodada"], st.session_state["analisado"] = lote.iloc[np.random.randint(len(lote))], 0, False
    if st.button("Sortear transação"):
        st.session_state["linha"] = lote.iloc[np.random.randint(len(lote))]
        st.session_state["rodada"] += 1
        st.session_state["analisado"] = False
    linha, rodada = st.session_state["linha"], st.session_state["rodada"]

    amount = st.number_input("Valor da transação (Amount)", min_value=0.0, value=float(linha["Amount"]),
                             step=10.0, key=f"amount_{rodada}")
    st.caption("As transações sorteadas vêm de um lote de demonstração com fraudes sobre-representadas\
                (cerca de 5%, contra 0,17% no dataset real).")
    with st.expander("Ver as variáveis V1 a V28"):
        colunas_v = [c for c in pacote["colunas"] if c.startswith("V")]
        st.dataframe(linha[colunas_v].astype(float).round(3).to_frame("valor"))
    if st.button("Analisar"):
        st.session_state["analisado"] = True
    if st.session_state.get("analisado"):
        mostrar_resultado(linha, amount, pacote)


def aba_ranking(pacote):
    """Aba 'Ranking de risco': lista transações em ordem de score e permite inspecionar qualquer posição.

    Duas fontes: o ranking REAL (as 300 mais suspeitas do teste inteiro, com a precisão verdadeira)
    e o lote de demonstração (fraudes sobre-representadas, só para mostrar casos).
    """
    fonte = st.radio("Fonte do ranking",
                     ["Top 300 do teste inteiro (real)", "Lote de demonstração (fraudes sobre-representadas)"],
                     horizontal=True)
    real = fonte.startswith("Top") and "ranking_real" in pacote
    lote = (pacote["ranking_real"] if real else pacote["lote"]).reset_index(drop=True)

    tabela = lote[["score", "Amount", "Time"]].copy()
    tabela.insert(0, "posição", range(1, len(tabela) + 1))
    ref = np.sort(pacote["scores_referencia"])
    percentis = np.searchsorted(ref, lote["score"].to_numpy()) / len(ref)
    tabela["decisão"] = np.where(lote["score"] >= pacote["limiar"], "IMPOSTOR",
                                 np.where(percentis >= 0.99, "atenção", "tripulante"))
    st.dataframe(tabela.head(15).round(3), hide_index=True)

    topo = lote.head(15)
    if real:
        alertas = lote[lote["score"] >= pacote["limiar"]]
        st.caption(f"Ranking real: entre as 15 primeiras, {int((topo['Class'] == 1).sum())} são fraudes de verdade. "
                   f"Nos {len(alertas)} alertas do topo, {int((alertas['Class'] == 1).sum())} são fraudes "
                   f"(precisão de {(alertas['Class'] == 1).mean():.1%}).")
    else:
        st.caption(f"Entre as 15 primeiras, {int((topo['Class'] == 1).sum())} são fraudes de verdade. "
                   f"Atenção: o lote tem {lote['Class'].mean():.1%} de fraudes (no dataset real são 0,17%), "
                   f"então essa proporção NÃO representa a precisão real do modelo.")

    escolha = st.number_input("Inspecionar a transação da posição", min_value=1, max_value=len(lote), value=1, step=1)
    linha = lote.iloc[int(escolha) - 1]
    mostrar_resultado(linha, float(linha["Amount"]), pacote)

RESULTADOS = Path("results")  # pasta gerada por: python -m src.experimentos --data data/creditcard.csv --e4


@st.cache_data
def carregar_resultados():
    """Lê os arquivos reais de results/ (nada de número digitado à mão).

    Devolve um dicionário com e1, e2, e3, e4, extra e run_info. O que não existir vira None,
    para a aba avisar em vez de quebrar o app.
    """
    def csv(nome):
        arq = RESULTADOS / nome
        return pd.read_csv(arq) if arq.exists() else None

    arq_info = RESULTADOS / "run_info.json"
    info = json.loads(arq_info.read_text(encoding="utf-8")) if arq_info.exists() else None
    return {"e1": csv("e1_resultados.csv"), "e2": csv("e2_resultados.csv"),
            "e3": csv("e3_resultados.csv"), "e4": csv("e4_resultados.csv"),
            "extra": csv("extra_sem_time.csv"), "info": info}


def matriz_do_e1(e1, info, metodo="Isolation Forest"):
    """Reconstrói a matriz de confusão do E1 a partir de precisão, recall e nº de alertas.

    tp = precisão x alertas; fraudes do teste = tp / recall; fp = alertas - tp;
    fn = fraudes - tp; tn = o resto do teste (tamanho do teste = ceil(0,3 x linhas sem duplicatas)).
    """
    linha = e1[e1["metodo"] == metodo].iloc[0]
    alertas = int(linha["n_alertas"])
    tp = round(linha["precisao"] * alertas)
    fraudes = round(tp / linha["recall"])
    fp, fn = alertas - tp, fraudes - tp
    n_teste = math.ceil(info["tamanho_teste"] * info["dados"]["linhas_apos_remover_duplicadas"])
    tn = n_teste - tp - fp - fn
    return np.array([[tn, fp], [fn, tp]]), alertas


def desenhar_matriz(matriz):
    """Heatmap 2x2 da matriz de confusão, só com matplotlib."""
    fig, eixo = plt.subplots(figsize=(4.5, 3))
    eixo.grid(False)
    eixo.imshow(np.log1p(matriz), cmap="Blues")
    for i in range(2):
        for j in range(2):
            eixo.text(j, i, f"{matriz[i, j]:,}".replace(",", "."), ha="center", va="center",
                      color="white" if matriz[i, j] > matriz.max() / 10 else "black", fontsize=12)
    eixo.set_xticks([0, 1], ["Normal", "Alerta"])
    eixo.set_yticks([0, 1], ["Real normal", "Real fraude"])
    eixo.set_xlabel("Decisão do modelo")
    eixo.set_ylabel("Classe real")
    fig.tight_layout()
    return fig


def aba_resultados():
    """Aba 'Resultados do modelo': tudo vem dos arquivos de results/ gerados pelo pipeline."""
    r = carregar_resultados()
    faltando = [k for k in ("e1", "e3", "e4", "info") if r[k] is None]
    st.header("Resultados dos experimentos e métricas")
    if r["info"] is None:
        st.error("Não achei results/. Rode: python -m src.experimentos --data data/creditcard.csv --e4")
        return
    if faltando:
        st.warning(f"Faltam arquivos em results/: {', '.join(faltando)}. Rode o pipeline com --e4.")

    e1, info = r["e1"], r["info"]
    prop = info["dados"]["proporcao_fraude"]
    sempre = e1[e1["metodo"] == "Sempre normal"]["acuracia"].iloc[0] if e1 is not None else 1 - prop

    with st.expander("Por que essas métricas?", expanded=True):
        st.markdown(f"""
**Acurácia engana aqui.** Fraude é {prop:.3%} das transações. Um modelo que diz "tudo normal" tem
**{sempre:.2%} de acurácia** e não pega nenhuma fraude (linha "Sempre normal" da tabela abaixo).

1. **PR-AUC (métrica principal):** foca na classe rara e não é inflada pelos verdadeiros negativos.
   Por isso o "sempre normal" tem PR-AUC ≈ {prop:.4f} (a taxa de fraude), e é essa a régua para comparar os métodos.
2. **Precisão e recall com orçamento fixo de alertas (@K):** a equipe de revisão tem capacidade limitada.
   Todos os métodos geram o mesmo número de alertas, então a comparação é justa.
3. **ROC-AUC (complementar):** mostra a ordenação global dos scores, mas fica otimista com classes muito desbalanceadas.
""")

    st.divider()
    st.subheader("E1: Isolation Forest vs baselines")
    if e1 is not None:
        tabela = e1[["metodo", "pr_auc", "roc_auc", "precisao", "recall", "f1", "n_alertas"]].copy()
        tabela.columns = ["Método", "PR-AUC", "ROC-AUC", "Precisão", "Recall", "F1", "Alertas"]
        st.dataframe(tabela, hide_index=True, width="stretch")
        st.caption("Precisão, recall e F1 dependem do limiar (mesmo nº de alertas para IF, Z-score e LOF); "
                   "PR-AUC e ROC-AUC não dependem. Uma única divisão treino/teste, com poucas fraudes no teste.")
    if (RESULTADOS / "e1_curva_pr.png").exists():
        st.image(str(RESULTADOS / "e1_curva_pr.png"), caption="Curva precisão x recall (dados reais do E1)")

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("E4: estabilidade (10 sementes)")
        e4 = r["e4"]
        if e4 is not None:
            z = float(e1[e1["metodo"] == "Z-score"]["pr_auc"].iloc[0]) if e1 is not None else None
            fig, eixo = plt.subplots(figsize=(5, 3.5))
            eixo.bar(e4["semente"].astype(str), e4["pr_auc"], color=AZUL, label="Isolation Forest")
            if z is not None:
                eixo.axhline(z, color=ATENCAO, linestyle="--", label=f"Z-score ({z:.3f})")
            eixo.set_xlabel("Semente")
            eixo.set_ylabel("PR-AUC")
            eixo.set_ylim(0, float(e4["pr_auc"].max()) * 1.3)
            eixo.legend(fontsize=8, loc="upper center", ncol=2)
            fig.tight_layout()
            st.pyplot(fig)
            ganhou = int((e4["pr_auc"] > z).sum()) if z is not None else 0
            st.caption(f"Média {e4['pr_auc'].mean():.3f} ± {e4['pr_auc'].std(ddof=0):.3f} "
                       f"(mín {e4['pr_auc'].min():.3f}, máx {e4['pr_auc'].max():.3f}); "
                       f"supera o Z-score em {ganhou} de {len(e4)} sementes.")
        else:
            st.info("E4 não encontrado. Rode o pipeline com --e4.")
    with col2:
        st.subheader("E3: efeito do limiar")
        if (RESULTADOS / "e3_grafico.png").exists():
            st.image(str(RESULTADOS / "e3_grafico.png"))
        if r["e3"] is not None:
            e3 = r["e3"][["contamination", "n_alertas", "precisao", "recall", "f1"]]
            st.dataframe(e3, hide_index=True, width="stretch")

    st.divider()
    if (RESULTADOS / "e2_grafico.png").exists():
        st.subheader("E2: parâmetros do Isolation Forest")
        st.image(str(RESULTADOS / "e2_grafico.png"))

    if r["extra"] is not None:
        st.subheader("Extra: sem a coluna Time")
        st.dataframe(r["extra"], hide_index=True, width="stretch")

    if e1 is not None:
        st.divider()
        matriz, alertas = matriz_do_e1(e1, info)
        st.subheader(f"Matriz de confusão do Isolation Forest ({alertas} alertas)")
        c_cm, c_txt = st.columns(2)
        with c_cm:
            st.pyplot(desenhar_matriz(matriz))
        with c_txt:
            (tn, fp), (fn, tp) = matriz
            st.markdown(f"""
- **Verdadeiros positivos ({tp}):** fraudes que entraram nos alertas.
- **Falsos positivos ({fp}):** alarmes falsos para a equipe revisar.
- **Falsos negativos ({fn}):** fraudes que ficaram fora dos alertas.
- **Verdadeiros negativos ({tn:,}):** transações normais não alertadas.
""".replace(",", "."))


def main():
    """Monta a página: título, texto de apresentação e as três abas."""
    pacote = carregar_pacote()
    # orçamento de alertas ajustável: o limiar vira o quantil dos scores do teste
    pct = st.sidebar.slider("Orçamento de alertas (% das transações)", 0.05, 5.0, 0.2, step=0.05)
    pacote = {**pacote, "limiar": float(np.quantile(pacote["scores_referencia"], 1 - pct / 100))}
    st.sidebar.caption("Valores do E3: 0,2% → recall 28%, precisão 24%; 0,5% → recall 44%, precisão 14%; "
                       "1% → recall 57%, precisão 9%.")
    st.title("Central de Impostores")
    st.write("Existe um impostor entre essas transações? O Isolation Forest isola quem foge do padrão. "
             "Protótipo experimental de triagem: o alerta não confirma fraude.")
    aba1, aba2, aba3 = st.tabs(["Analisar transação", "Ranking de risco", "Resultados do modelo"])
    with aba1:
        aba_analisar(pacote)
    with aba2:
        aba_ranking(pacote)
    with aba3:
        aba_resultados()


main()

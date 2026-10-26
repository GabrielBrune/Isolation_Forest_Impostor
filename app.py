# Protótipo "Central de Impostores" - Isolation Forest
# Antes de abrir, gere o modelo:  python -m src.experimentos --data data/creditcard.csv
# Para abrir:                      python -m streamlit run app.py

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Central de Impostores", page_icon="🕵️", layout="wide")


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
    eixo.scatter(normais["x"], normais["y"], s=6, c="lightgray", label="Tripulantes")
    eixo.scatter(suspeitos["x"], suspeitos["y"], s=14, c="salmon", label="Impostores (modelo)")
    eixo.scatter([ponto[0]], [ponto[1]], s=220, c="red" if impostor else "green",
                 edgecolors="black", marker="*", label="Transação analisada")
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
        st.error("🚨 IMPOSTOR! Essa transação foge do padrão. (Alerta para revisão, não confirma fraude.)")
    elif percentil >= 0.99:
        st.warning("⚠️ Tripulante, mas na zona de atenção: o score ficou logo abaixo do limiar.")
    else:
        st.success("✅ Tripulante. Essa transação parece normal.")
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
    if st.button("SORTEAR TRANSAÇÃO"):
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
    if st.button("ANALISAR"):
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

def main():
    """Monta a página: título, texto de apresentação e as duas abas."""
    pacote = carregar_pacote()
    # orçamento de alertas ajustável: o limiar vira o quantil dos scores do teste
    pct = st.sidebar.slider("Orçamento de alertas (% das transações)", 0.05, 5.0, 0.2, step=0.05)
    pacote = {**pacote, "limiar": float(np.quantile(pacote["scores_referencia"], 1 - pct / 100))}
    st.sidebar.caption("Valores do E3: 0,2% → recall 28%, precisão 24%; 0,5% → recall 44%, precisão 14%; "
                       "1% → recall 57%, precisão 9%.")
    st.title("🕵️ Central de Impostores")
    st.write("Existe um impostor entre essas transações? O Isolation Forest isola quem foge do padrão. "
             "Protótipo experimental de triagem: o alerta não confirma fraude.")
    aba1, aba2 = st.tabs(["Analisar transação", "Ranking de risco"])
    with aba1:
        aba_analisar(pacote)
    with aba2:
        aba_ranking(pacote)


main()

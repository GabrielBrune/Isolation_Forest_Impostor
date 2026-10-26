"""Funções compartilhadas: leitura dos dados, preparação, scores, limiares e métricas."""
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

# ---------- configuração única (README e resumo devem citar estes valores) ----------
ARQUIVO_PADRAO = "data/creditcard.csv"
SEMENTE = 42
TAMANHO_TESTE = 0.3          # divisão 70/30, estratificada
CONTAMINACAO = 0.002         # orçamento de alertas: 0,2% das transações (a fraude real é ~0,17%)
LOF_VIZINHOS = 20


# ===================== DADOS =====================

def carregar_dados(caminho=ARQUIVO_PADRAO):
    """Lê o CSV, confere a qualidade e remove as linhas duplicadas.

    Por que existe: linhas idênticas poderiam cair no treino e no teste ao mesmo
    tempo, o que inflaria o resultado (vazamento de dados). Por isso removemos e contamos.
    Entrada: caminho do CSV.
    Saída: (df, info). `df` é o dataset limpo; `info` é um dicionário com as
    contagens (linhas, duplicadas, fraudes, ausentes) que vão para o relatório.
    """
    caminho = Path(caminho)
    if not caminho.exists():
        raise SystemExit(f"Não achei {caminho}. Baixe o dataset (ver data/README.md) e salve nesse caminho.")
    df = pd.read_csv(caminho)
    if "Class" not in df.columns:
        raise ValueError("O dataset precisa ter a coluna-alvo 'Class'.")
    if df["Class"].isna().any() or not set(df["Class"].unique()).issubset({0, 1}):
        raise ValueError("A coluna Class deve conter apenas 0 (normal) e 1 (fraude).")

    duplicada = df.duplicated()
    info = {
        "linhas_originais": int(len(df)),
        "colunas": int(df.shape[1]),
        "valores_ausentes": int(df.isna().sum().sum()),
        "linhas_duplicadas": int(duplicada.sum()),
        "fraudes_entre_duplicadas": int(df.loc[duplicada, "Class"].sum()),
        "fraudes_originais": int(df["Class"].sum()),
    }
    # política: remover duplicatas (linhas idênticas poderiam cair em treino e teste ao mesmo tempo)
    df = df.drop_duplicates()
    info["linhas_apos_remover_duplicadas"] = int(len(df))
    info["fraudes_apos_remover_duplicadas"] = int(df["Class"].sum())
    info["proporcao_fraude"] = float(df["Class"].mean())
    return df, info


@dataclass
class Dados:
    X_treino: pd.DataFrame      # já padronizado (scaler ajustado só no treino)
    X_teste: pd.DataFrame
    y_treino: pd.Series
    y_teste: pd.Series
    scaler: StandardScaler
    colunas: list
    linhas_repetidas: int       # linhas idênticas entre treino e teste (deve ser 0)


def preparar(df, sem_time=False, semente=SEMENTE, tamanho_teste=TAMANHO_TESTE, escalonador="padrao"):
    """Divide em treino/teste e padroniza as variáveis, sem vazamento.

    Passos:
     1) separa 70% treino / 30% teste, estratificado (mesma proporção de fraudes nos dois);
     2) confere se sobrou alguma linha idêntica nos dois lados (deve dar 0);
     3) preenche valores ausentes com a mediana do TREINO (só age se houver ausentes);
     4) padroniza (média 0, desvio 1) usando só o TREINO e aplica no teste.
    A coluna `Class` nunca entra nas variáveis: o modelo não vê o rótulo.
    `sem_time=True` retira a coluna Time (usado na ablação).
    Saída: objeto `Dados` com treino, teste, rótulos, scaler e a contagem de repetidas.
    """
    colunas = [c for c in df.select_dtypes(include="number").columns if c != "Class"]
    if sem_time:
        colunas = [c for c in colunas if c != "Time"]
    X, y = df[colunas], df["Class"].astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=tamanho_teste, stratify=y, random_state=semente)

    # checagem de vazamento: nenhuma linha idêntica no treino e no teste
    hash_tr = set(pd.util.hash_pandas_object(X_tr, index=False))
    hash_te = set(pd.util.hash_pandas_object(X_te, index=False))
    repetidas = len(hash_tr & hash_te)

    medianas = X_tr.median().fillna(0)       # só age se houver valores ausentes
    X_tr, X_te = X_tr.fillna(medianas), X_te.fillna(medianas)

    # fit SÓ no treino: o teste é transformado com a média/desvio do treino (evita vazamento)
    scaler = (StandardScaler()).fit(X_tr)
    X_tr = pd.DataFrame(scaler.transform(X_tr), columns=colunas, index=X_tr.index)
    X_te = pd.DataFrame(scaler.transform(X_te), columns=colunas, index=X_te.index)
    return Dados(X_tr, X_te, y_tr, y_te, scaler, colunas, repetidas)


# ===================== SCORES (maior = mais anômalo) =====================

def ajustar_zscore_robusto(X_treino):
    """Versão robusta do Z-score: usa mediana e MAD no lugar de média e desvio-padrão.

    MAD = mediana dos desvios absolutos em relação à mediana (multiplicado por 1,4826
    para ficar comparável ao desvio-padrão). Valores extremos quase não afetam esses
    dois números, ao contrário da média e do desvio-padrão.
    Saída: (medianas, mad), que entram em `pontuar_zscore` no lugar de (médias, desvios).
    """
    medianas = X_treino.median(axis=0)
    mad = (X_treino - medianas).abs().median(axis=0) * 1.4826
    return medianas, mad.replace(0, 1).fillna(1)

def ajustar_zscore(X_treino):
    """Parte 'treinar' do Z-score: aprende a média e o desvio-padrão de cada coluna no treino.

    Se uma coluna tiver desvio zero, usamos 1 para evitar divisão por zero.
    Saída: (médias, desvios).
    """
    medias = X_treino.mean(axis=0)
    desvios = X_treino.std(axis=0, ddof=0).replace(0, 1).fillna(1)
    return medias, desvios


def pontuar_zscore(X, medias, desvios):
    """Score do baseline: o quanto cada transação foge do normal em ALGUMA variável.

    Para cada transação calcula |z| = |valor - média| / desvio em todas as colunas
    e fica com o MAIOR valor. Quanto maior o score, mais suspeita a transação.
    Entrada: dados X e as médias/desvios do treino. Saída: um score por linha.
    """
    return ((X - medias) / desvios).abs().max(axis=1).to_numpy()


def treinar_if(X_treino, n_estimators=100, max_samples="auto", semente=SEMENTE):
    """Treina o Isolation Forest sem usar rótulos e mede o tempo de treino.

    n_estimators: quantas árvores de isolamento a floresta tem.
    max_samples: quantos pontos cada árvore sorteia ('auto' = 256).
    O tempo de treino depende desses dois parâmetros, e não do total de linhas.
    Saída: (modelo treinado, segundos de treino).
    """
    modelo = IsolationForest(n_estimators=n_estimators, max_samples=max_samples,
                             random_state=semente, n_jobs=-1)
    inicio = time.perf_counter()
    modelo.fit(X_treino)
    return modelo, time.perf_counter() - inicio


def pontuar_if(modelo, X):
    # o scikit-learn devolve valores MAIORES para pontos normais; invertemos o sinal
    """Score de anomalia do Isolation Forest: maior = mais suspeito.

    O scikit-learn devolve valores MAIORES para pontos normais; trocamos o sinal para
    que o sentido seja igual ao do Z-score e do LOF. Esquecer essa inversão faz a
    PR-AUC sair invertida.
    """
    return -modelo.score_samples(X)


def pontuar_lof(X_treino, X_teste, max_treino=20000, semente=SEMENTE):
    """Score do Local Outlier Factor (LOF), nosso segundo método de comparação.

    O LOF compara a densidade ao redor de um ponto com a dos seus vizinhos. Usamos
    `novelty=True` para poder pontuar dados novos (o teste). Como o custo cresce muito
    com o tamanho dos dados, o treino é uma subamostra de `max_treino` linhas.
    Medimos os dois tempos porque o custo do LOF está sobretudo na PREDIÇÃO.
    Saída: (scores do teste, segundos de fit, segundos de predição).
    """
    if max_treino and len(X_treino) > max_treino:
        X_treino = X_treino.sample(max_treino, random_state=semente)
    # arrays numpy evitam o aviso de nomes de colunas do scikit-learn
    treino, teste = X_treino.to_numpy(), X_teste.to_numpy()
    lof = LocalOutlierFactor(n_neighbors=LOF_VIZINHOS, novelty=True, n_jobs=-1)
    inicio = time.perf_counter()
    lof.fit(treino)
    t_fit = time.perf_counter() - inicio
    inicio = time.perf_counter()
    scores = -lof.score_samples(teste)
    return scores, t_fit, time.perf_counter() - inicio


# ===================== LIMIARES E MÉTRICAS =====================

def limiar_orcamento(scores, fracao):
    """Limiar que deixa como alerta as `fracao` transações mais suspeitas do conjunto.

    É o 'orçamento de alertas': todos os métodos geram o mesmo número de alertas, o que
    torna precisão, recall e F1 comparáveis entre eles.
    Exemplo: fracao=0,002 em 85 mil transações dá cerca de 170 alertas.
    """
    k = max(1, int(round(fracao * len(scores))))
    return float(np.partition(scores, -k)[-k])


def limiar_por_quantil(scores_treino, contaminacao):
    """Limiar operacional do protótipo, calculado só com os scores do TREINO.

    Pega o quantil (1 - contaminação) dos scores de treino. Vale para transações novas
    (como as que o usuário digita no app), pois não depende de olhar o teste.
    """
    return float(np.quantile(scores_treino, 1 - contaminacao))


def calcular_metricas(y_real, scores, limiar):
    """Calcula as métricas de um detector para um limiar dado.

    PR-AUC e ROC-AUC usam o score inteiro e NÃO dependem do limiar.
    Precisão, recall, F1, acurácia e n_alertas dependem do limiar: viram 'alerta'
    as transações com score >= limiar.
    Entrada: rótulos reais, scores, limiar. Saída: dicionário com as métricas.
    """
    previsto = (scores >= limiar).astype(int)
    return {
        "pr_auc": float(average_precision_score(y_real, scores)),
        "roc_auc": float(roc_auc_score(y_real, scores)),
        "precisao": float(precision_score(y_real, previsto, zero_division=0)),
        "recall": float(recall_score(y_real, previsto, zero_division=0)),
        "f1": float(f1_score(y_real, previsto, zero_division=0)),
        "acuracia": float(accuracy_score(y_real, previsto)),
        "n_alertas": int(previsto.sum()),
    }

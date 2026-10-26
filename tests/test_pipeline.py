import numpy as np
import pandas as pd

from src.common import (ajustar_zscore, calcular_metricas, carregar_dados, limiar_orcamento,
                        limiar_por_quantil, pontuar_if, pontuar_zscore, preparar, treinar_if)


def df_sintetico(n=4000, frac_fraude=0.02, semente=0):
    """Dados sintéticos no formato do dataset, só para testar o código (nunca para resultados)."""
    rng = np.random.RandomState(semente)
    df = pd.DataFrame(rng.normal(size=(n, 28)), columns=[f"V{i}" for i in range(1, 29)])
    df.insert(0, "Time", rng.uniform(0, 172800, n))
    df["Amount"] = rng.lognormal(3, 1, n)
    df["Class"] = 0
    fraudes = rng.choice(n, int(n * frac_fraude), replace=False)
    df.loc[fraudes, [f"V{i}" for i in range(1, 8)]] += 6
    df.loc[fraudes, "Class"] = 1
    return df


def test_zscore_nao_negativo_e_tamanho():
    treino = pd.DataFrame({"a": [0.0, 1.0, 2.0], "b": [1.0, 1.0, 1.0]})
    teste = pd.DataFrame({"a": [1.0, 10.0], "b": [1.0, 2.0]})
    medias, desvios = ajustar_zscore(treino)
    scores = pontuar_zscore(teste, medias, desvios)
    assert len(scores) == 2 and np.all(scores >= 0)
    assert scores[1] > scores[0]


def test_limiar_orcamento_gera_k_alertas():
    scores = np.arange(1, 101, dtype=float)
    limiar = limiar_orcamento(scores, 0.10)
    assert (scores >= limiar).sum() == 10


def test_limiar_por_quantil():
    scores = np.arange(1, 1001, dtype=float)
    assert (scores >= limiar_por_quantil(scores, 0.01)).sum() in (10, 11)


def test_duplicatas_removidas(tmp_path):
    df = df_sintetico(500)
    pd.concat([df, df.iloc[:20]]).to_csv(tmp_path / "d.csv", index=False)
    limpo, info = carregar_dados(tmp_path / "d.csv")
    assert info["linhas_duplicadas"] == 20 and len(limpo) == 500


def test_preparacao_sem_vazamento_e_sem_rotulo():
    d = preparar(df_sintetico())
    assert "Class" not in d.colunas
    assert d.linhas_repetidas == 0
    assert np.allclose(d.X_treino.mean().to_numpy(), 0, atol=1e-8)   # scaler aprendido no treino
    assert abs(d.y_treino.mean() - d.y_teste.mean()) < 0.005          # estratificado


def test_isolation_forest_acha_anomalias_sinteticas():
    d = preparar(df_sintetico())
    modelo, _ = treinar_if(d.X_treino)
    scores = pontuar_if(modelo, d.X_teste)
    m = calcular_metricas(d.y_teste.to_numpy(), scores, limiar_orcamento(scores, 0.02))
    assert m["pr_auc"] > 0.5            # bem acima da prevalência (~0,02)
    assert 0 <= m["recall"] <= 1

# Checkpoint 2 (03/11): interpretação dos resultados

> Rascunho com os números reais da execução no `creditcard.csv` completo. O grupo precisa ler, concordar e reescrever com as próprias palavras. As hipóteses são as do README (seção 3), registradas antes de rodar. Os resultados vêm de **uma única divisão treino/teste** (o E4 mede a variação entre 10 sementes do algoritmo), então diferenças pequenas não são conclusivas.

## Resumo: o que cada hipótese mostrou

| Hipótese | Veredito | Em uma frase |
|---|---|---|
| H1 | Sustentada, com ressalvas | Isolation Forest tem a maior PR-AUC (0,1335 contra 0,0780 do Z-score e 0,0331 do LOF) e supera o Z-score em 9 das 10 sementes, mas a média das sementes é 0,108. |
| H2 | Parcialmente confirmada | Tempo depende dos parâmetros e não dos registros, e o custo do LOF está na predição; mas com `max_samples` pequeno a PR-AUC **cai**. |
| H3 | Confirmada | `contamination` só move o limiar (PR-AUC fixa em 0,1335), recall sobe, precisão cai, e a acurácia continua alta com recall baixo. |
| Extra (Time) | Sem evidência de efeito | Sem `Time` a PR-AUC vai de 0,1335 para 0,1238, diferença dentro da variação entre sementes. |
| H4 (opcional) | Não sustentada | O resultado varia bastante entre sementes: de 0,043 a 0,136 (média 0,108, desvio 0,025). |

## Dados usados

- 283.726 transações após remover 1.081 duplicatas (19 delas eram fraudes), com 473 fraudes (0,167%).
- Nenhuma linha repetida entre treino e teste.
- O teste tem cerca de 85 mil transações e 142 fraudes (30% da base, divisão estratificada).

## E1: Isolation Forest × Z-score × LOF

**Hipótese (H1):** o Isolation Forest tem PR-AUC maior que o Z-score e o LOF.

| Método | PR-AUC | ROC-AUC | Precisão | Recall | F1 | Acurácia |
|---|---|---|---|---|---|---|
| Isolation Forest | 0,1335 | 0,9447 | 0,2353 | 0,2817 | 0,2564 | 0,9973 |
| Z-score | 0,0780 | 0,9321 | 0,1294 | 0,1549 | 0,1410 | 0,9969 |
| LOF | 0,0331 | 0,8800 | 0,0706 | 0,0845 | 0,0769 | 0,9966 |
| Sempre normal | 0,0017 | 0,5000 | 0 | 0 | 0 | 0,9983 |

Os três métodos usaram o mesmo número de alertas (170).

**Leitura:** H1 se confirma. O Isolation Forest tem a maior PR-AUC (0,1335 contra 0,0780 do Z-score e 0,0331 do LOF), e acerta 40 das 142 fraudes entre 170 alertas. O LOF foi o pior método e também o mais lento na predição (de 1,2 a 2,6 s conforme a execução, contra cerca de 0,4 s do Isolation Forest). Mesmo assim, o desempenho absoluto é modesto: a PR-AUC de 0,13 fica cerca de 80 vezes acima do nível aleatório (0,0017), mas a precisão é de 23% e o recall de 28%.

**Um contraste importante:** a ROC-AUC do Isolation Forest é alta (0,94) enquanto a PR-AUC é baixa (0,13). É o que o artigo de Saito e Rehmsmeier (2015) descreve: com classes muito desbalanceadas, a ROC-AUC parece boa e esconde a quantidade de falsos alarmes.

**Z-score com limiar fixo (|z| ≥ 3):** pega 87% das fraudes, mas dispara 11.546 alertas, com precisão de 1,07% e acurácia de 86,6%. Um limiar fixo cria uma fila de revisão inviável.

## E2: n_estimators, max_samples e tamanho do treino

**Hipótese (H2):** o tempo de treino do Isolation Forest depende de `n_estimators` e `max_samples`, e não do número de registros; o custo do LOF está sobretudo na predição; com `max_samples` pequeno a PR-AUC se mantém próxima.

| Fator | Variação | Efeito no tempo de treino | Efeito na PR-AUC |
|---|---|---|---|
| `n_estimators` | 25 a 400 | de 0,11 s para 1,41 s (cerca de 12 vezes, quase linear) | oscila entre 0,102 e 0,134, sem tendência clara |
| `max_samples` | 64 a 2048 | de 0,32 s para 0,81 s (cerca de 2,5 vezes) | sobe de 0,071 para 0,176 |
| registros de treino | 19.861 a 198.608 | de 0,19 s para 0,39 s (10 vezes mais dados, cerca de 2 vezes mais tempo) | oscila entre 0,121 e 0,138, sem tendência |

**Leitura, parte por parte da H2:**
- **Tempo depende de `n_estimators` e `max_samples`, não dos registros: confirmada.** O tempo cresce de forma quase proporcional ao número de árvores (cerca de 12 vezes para 16 vezes mais árvores), cresce pouco com `max_samples` e quase nada com o tamanho do treino (10 vezes mais dados, cerca de 2 vezes mais tempo, e entre 49.652 e 99.304 registros ele não aumentou). Isso acontece porque cada árvore usa só uma subamostra.
- **Custo do LOF está na predição: confirmada.** O LOF levou cerca de 0,7 s para treinar e de 1,2 a 2,6 s para prever (conforme a execução), contra cerca de 0,35 s e 0,4 s do Isolation Forest.
- **Com `max_samples` pequeno a PR-AUC se mantém: não confirmada.** Com 64 amostras por árvore a PR-AUC cai para 0,071 (abaixo do Z-score, 0,078), e com 2048 sobe para 0,176. Quanto maior o `max_samples`, melhor, ao menos nesse intervalo.
- Mais árvores não melhoraram a PR-AUC de forma consistente: as variações de 0,02 a 0,03 provavelmente são ruído (o E4 mostra um desvio de 0,025 só por trocar a semente), e `n_estimators=100` ter sido o maior pode ser acaso.
- Em todas as outras configurações o Isolation Forest fica acima do Z-score, o que reforça a conclusão do E1.

## E3: limiar (`contamination`)

**Hipótese (H3):** `contamination` só move o limiar (o ranking dos scores não muda); aumentá-lo sobe o recall e derruba a precisão; a acurácia continua alta mesmo com recall baixo.

| contamination | Alertas | Precisão | Recall | F1 | Acurácia |
|---|---|---|---|---|---|
| 0,0005 | 41 | 0,2927 | 0,0845 | 0,1311 | 0,9981 |
| 0,001 | 84 | 0,2976 | 0,1761 | 0,2212 | 0,9979 |
| **0,002** | 165 | 0,2424 | 0,2817 | **0,2606** | 0,9973 |
| 0,005 | 443 | 0,1400 | 0,4366 | 0,2120 | 0,9946 |
| 0,01 | 916 | 0,0884 | 0,5704 | 0,1531 | 0,9895 |
| 0,02 | 1.738 | 0,0575 | 0,7042 | 0,1064 | 0,9803 |
| 0,05 | 4.278 | 0,0266 | 0,8028 | 0,0516 | 0,9508 |

A PR-AUC é sempre 0,1335, porque não depende do limiar.

**Leitura, parte por parte da H3:**
- **`contamination` só move o limiar: confirmada.** A PR-AUC é igual em todas as linhas (0,1335), então o ranking não mudou.
- **Recall sobe e precisão cai: confirmada, com uma exceção pequena.** O recall cresce em todas as linhas (de 0,085 para 0,803). A precisão cai de 0,298 (com 0,001) para 0,027 (com 0,05), com uma subida mínima no começo (0,293 com 0,0005 para 0,298 com 0,001).
- **A acurácia continua alta mesmo com recall baixo: confirmada.** A acurácia mais alta (0,9981) aparece onde o recall é de apenas 8%.

**Observação adicional:** mesmo no melhor ponto de F1 (0,002, valor próximo da proporção real de fraude) a acurácia, 0,9973, fica abaixo da do modelo que nunca aponta fraude (0,9983). Ou seja, a acurácia chega a premiar o modelo que não faz nada. O `contamination` é uma decisão de negócio:

- com 0,01, o analista revisaria 916 transações para achar cerca de 81 das 142 fraudes;
- com 0,05, seriam 4.278 transações para achar cerca de 114.

## E4 (opcional): estabilidade com 10 sementes

**Hipótese (H4):** o resultado varia pouco entre sementes aleatórias diferentes.

PR-AUC média de **0,108**, desvio-padrão de **0,025**, mínimo de 0,043 (semente 8) e máximo de 0,136 (semente 6). O Isolation Forest supera o Z-score em 9 das 10 sementes.

**Leitura:** a H4 **não se sustenta**: a PR-AUC varia bastante com a semente. O 0,1335 do E1 (semente 42) fica perto do topo dessa faixa e deve ser lido como a execução principal, e não como valor típico. Isso também relativiza diferenças pequenas nos outros experimentos (`n_estimators`, `Time`), que ficam dentro desse ruído.

## Métricas justificadas

- **PR-AUC:** principal métrica de comparação entre métodos, porque não depende do limiar e não se infla com os muitos casos normais (Saito e Rehmsmeier, 2015).
- **Precisão, recall e F1:** mostram o custo e o ganho de um limiar escolhido (quantos alertas são verdadeiros e quantas fraudes escapam).
- **Acurácia:** só para mostrar o contraste, pois 99,83% é o que se obtém sem detectar nada.
- **Tempo de treino e de predição:** o custo prático de cada método.

## Análise de erros (detalhes em `analise_de_erros.csv`)

- **Falsos positivos (3 casos):** as transações mais "impostoras" do teste têm Amount de 11.898 a 25.691 (a mediana do dataset é 22) e variáveis V extremas. IF e LOF as colocam nas mesmas três primeiras posições. São pontos realmente isolados, mas não são fraude: o modelo acha o raro, não o fraudulento.
- **Falsos negativos (3 casos):** fraudes de valor muito baixo (0,76 a 1,18) e variáveis V próximas do normal. O IF as coloca por volta das posições 56 mil a 70 mil, de cerca de 85 mil. Hipótese: fraudes camufladas como compras pequenas comuns. Valores muito baixos podem ser testes de cartão, mas isso não foi verificado.
- **Fraudes que o LOF ranqueia melhor (3 casos):** uma delas fica na posição 85 no LOF e 30.703 no IF. Hipótese: fraudes isoladas localmente, mas não extremas no conjunto, que o LOF (densidade local) enxerga e o IF (cortes aleatórios globais) não. Sugere que os dois métodos se complementam.

## Extra: o efeito da variável Time

**Hipótese:** `Time` (segundos desde a primeira transação) não é comportamento do cartão, e retirá-lo não deve piorar a PR-AUC de forma relevante.

Sem `Time` a PR-AUC é 0,1238, e com `Time` é 0,1335 (queda de 0,0097). A diferença fica dentro da variação entre sementes (E4), então **não há evidência de efeito** nem a favor nem contra a hipótese. Como `Time` só cobre dois dias, a variável pode não generalizar, e vale mencionar isso como limitação.

## Observações da EDA que ajudam a interpretar

- 80% a 87% das fraudes são outliers (regra do IQR) em V17, V14 e V12, contra 2% a 5% das transações normais. Já no Amount são 18% contra 11%, um sinal fraco.
- A mediana do Amount nas fraudes (9,82) é menor que a das normais (22,00): as fraudes tendem a ter valor baixo, o contrário do que o modelo marca como "estranho" nos falsos positivos.
- Hipótese de trabalho (não testada): as fraudes são extremas em poucas variáveis (V10, V12, V14, V17), mas o Isolation Forest corta variáveis sorteadas ao acaso entre as 30, e boa parte dos cortes cai em variáveis sem informação. Isso pode limitar o desempenho.

## Conclusões provisórias

1. O Isolation Forest, sem rótulos no treino, supera o baseline estatístico e o LOF em PR-AUC (0,1335, 0,0780 e 0,0331 na semente 42), com custo computacional baixo. Na média de 10 sementes a PR-AUC é 0,108 ± 0,025, e ele supera o Z-score em 9 das 10. A qualidade depende de `max_samples`: com valores pequenos ele perde para o Z-score.
2. O desempenho absoluto é modesto: no ponto de F1 máximo, cerca de 1 alerta em cada 4 é fraude, e cerca de 3 em cada 10 fraudes são encontradas.
3. A acurácia não serve como métrica neste problema, e a escolha do limiar é uma decisão operacional que troca falsos alarmes por fraudes encontradas.
4. O modelo encontra o que é raro (compras de valor muito alto), e não necessariamente o que é fraude.

## Limitações

- Uma única divisão treino/teste, com só 142 fraudes no teste. O E4 mede a variação entre sementes do algoritmo, mas não a variação causada pela divisão.
- O `contamination` ótimo (0,002) foi observado com os rótulos do teste, e em uso real a proporção de fraude seria desconhecida.
- As variáveis V1 a V28 são anonimizadas e vêm de um PCA, então não dá para explicar o que cada uma significa.
- O dataset cobre apenas dois dias de transações europeias em 2013.
- O protótipo usa transações do próprio dataset, sem dados em tempo real.

## Próximos passos possíveis

- Repetir o E1 com várias divisões treino/teste, para medir a variação causada pela divisão.
- Testar uma combinação simples dos rankings do Isolation Forest e do LOF.
- Escolher variáveis informativas só como experimento exploratório, deixando claro que isso usaria os rótulos.

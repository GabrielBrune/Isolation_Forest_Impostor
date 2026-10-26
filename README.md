# Central de Impostores: Isolation Forest para detecção de fraudes em cartão

**Disciplina:** Inteligência Artificial II • 2026 • Grupo 6 • Tema 6: *Existe um impostor aqui?*
**Repositório:** https://github.com/GabrielBrune/Isolation_Forest_Impostor

## 1. Problema
Instituições financeiras recebem grandes volumes de transações e precisam priorizar as potencialmente fraudulentas para revisão. Como a fraude é rara (0,167% dos dados), a acurácia isolada engana: um sistema que nunca aponta fraude já "acerta" 99,83%. O projeto investiga se um método de detecção de anomalias, treinado **sem rótulos**, consegue priorizar fraudes melhor que uma regra estatística simples.

## 2. Pergunta de investigação
Sem usar rótulos no treino, o Isolation Forest apresenta PR-AUC superior à do baseline Z-score e à do Local Outlier Factor (LOF) no mesmo split de avaliação, e como o limiar de decisão equilibra falsos alarmes e fraudes não detectadas?

## 3. Hipóteses e o que os dados mostraram
As hipóteses foram registradas antes de rodar com o dataset real. A coluna de status resume o que os resultados da seção 9 sustentam **nesta divisão e com esta semente**.

| Hipótese | Status | Evidência |
|---|---|---|
| **H1:** o Isolation Forest tem PR-AUC maior que o Z-score e o LOF | **Sustentada, com ressalvas** | 0,1335 (semente 42) contra 0,0780 e 0,0331; média de 10 sementes 0,108 ± 0,025; supera o Z-score em 9 das 10 sementes. O LOF usa só 10% do treino |
| **H2:** o tempo de treino depende de `n_estimators` e `max_samples`, e não do número de registros; o custo do LOF está na predição; com `max_samples` pequeno a PR-AUC se mantém | **Parcial** | Treino quase linear em `n_estimators`; com 10× mais registros o tempo variou pouco (de 0,12 a 0,39 s entre execuções), muito menos que 10×. No LOF, predição (1,17 s) pesou mais que o fit (0,70 s). **Refutada** para `max_samples`: 64 → 0,071 e 128 → 0,104 contra 0,134 com 256 |
| **H3:** `contamination` só move o limiar; aumentá-lo sobe o recall e derruba a precisão; a acurácia continua alta | **Confirmada** | PR-AUC idêntica (0,1335) nas 7 linhas; recall de 8% a 80% e precisão de 29% a 2,7% |
| **H4 (opcional):** o resultado varia pouco entre sementes | **Não sustentada** | 0,043 a 0,136 entre 10 sementes (desvio de 0,025 sobre média de 0,108) |
| **Extra:** retirar `Time` não piora a PR-AUC de forma relevante | **Sem evidência de efeito** | 0,1335 com `Time`, 0,1238 sem; diferença dentro da variação entre sementes |
| **Extra 2 (registrada antes de rodar):** o Isolation Forest não muda com o escalonador (cada corte sorteia variável e ponto entre mínimo e máximo, então reescalar uma variável não altera a estrutura); Z-score robusto (mediana/MAD) e LOF com `RobustScaler` podem mudar | **Pendente** | [PREENCHER após rodar: `results/extra_robusto.csv`] |

## 4. Dataset e fonte
**Credit Card Fraud Detection** (Machine Learning Group da ULB), no Kaggle: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

- Transações de portadores europeus ao longo de dois dias em setembro de 2013.
- Colunas: `Time`, `Amount`, `V1` a `V28` (componentes de PCA, anonimizadas) e `Class` (1 = fraude; usada **só na avaliação**).
- **Contagens confirmadas na execução:** 284.807 transações e 31 colunas, nenhum valor ausente, 492 fraudes (0,173%). Foram removidas 1.081 linhas duplicadas (19 eram fraudes), restando 283.726 linhas e 473 fraudes (0,167%).
- Licença: [PREENCHER: copiar da página do dataset].
- O arquivo **não está no repositório** (150 MB; o limite do GitHub é 100 MB). Veja `data/README.md`.

## 5. Decisões de método
| Decisão | Valor | Por quê |
|---|---|---|
| Duplicatas | removidas e contadas (1.081) | linhas idênticas poderiam cair em treino e teste (vazamento); após a limpeza, 0 linhas idênticas entre treino e teste |
| Split | 70/30, estratificado, semente 42 (treino 198.608; teste 85.118, com 142 fraudes) | mantém fraudes no teste |
| Padronização | todas as colunas, ajustada só no treino | LOF depende de distância; evita vazamento |
| Imputação | mediana do treino | só atua se houver ausentes (não houve) |
| Rótulo `Class` | nunca entra no treino | detecção não supervisionada |
| Outliers | **não removidos** | são o alvo da detecção: 80% a 87% das fraudes são outliers pelo IQR em V12, V14 e V17, contra 2% a 5% das normais. Mas ser outlier não basta (11,2% das normais também são em `Amount`), por isso tratamos a anomalia como score contínuo e avaliamos em ranking |
| Baseline | Z-score: maior \|z\| entre as variáveis | regra simples de referência |
| Comparação P/R/F1 | **mesmo orçamento de alertas** (0,2% do teste, 170 alertas) para os três métodos | limiar fixo diferente por método deixaria a comparação injusta |
| Limiar do protótipo e do E3 | quantil dos scores de **treino** | vale para transações novas; por isso o E3 gera 165 alertas com `contamination` 0,002, e não 170 |
| LOF | `novelty=True`, 20 vizinhos, subamostra de 20 mil do treino (cerca de 10%) | custo computacional; limitação declarada |
| `Time` | mantida no pipeline principal; ablação à parte | não é comportamento do cartão, ver "Extra" |

## 6. Métricas
**PR-AUC** (average precision) é a principal e não depende do limiar. Também: ROC-AUC, precisão, recall, F1, acurácia (só como contraste), número de alertas, **tempo de treino e de predição**. A escolha do limiar muda precisão, recall e F1; parâmetros não devem ser escolhidos olhando o teste. A ROC-AUC alta (0,94) convive com PR-AUC baixa (0,13): é o motivo de preferirmos a curva precisão-recall em dados desbalanceados (SAITO; REHMSMEIER, 2015).

## 7. Estrutura
```text
src/common.py        dados, preparação, scores, limiares e métricas
src/eda.py           qualidade dos dados, 3 descobertas, outliers, checagem de vazamento
src/experimentos.py  E1 a E4, ablação de Time, análise de erros, modelo do protótipo
app.py               protótipo Streamlit (Central de Impostores)
tests/               testes automatizados (dados sintéticos, só para validar o código)
data/                dataset local (não versionado)
results/             saídas da execução (versionadas, como evidência)
docs/                resumo expandido, referências, checklist e diário de IA
```

## 8. Como executar
Testado no Windows com **Python 3.14.5** e as versões fixadas em `requirements.txt` (scikit-learn 1.9.1, pandas 3.1.0, numpy 2.5.4, streamlit 1.66.1). Em outra versão do Python pode ser necessário ajustar as bibliotecas. No Windows use `py` (em Mac/Linux, `python3`), na raiz do projeto:

```
py -m venv .venv
.venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt
py -m src.eda --data data/creditcard.csv
py -m src.experimentos --data data/creditcard.csv --e4
py -m streamlit run app.py
py -m pytest
```

- `src.experimentos` leva alguns minutos por causa do LOF. Use `--sem-lof` para pular, `--e4` para rodar o experimento das 10 sementes e `--lof-treino-max N` para mudar a subamostra do LOF.
- As tabelas (CSV), os gráficos (PNG) e o `run_info.json` (versões, parâmetros e contagens) ficam em `results/`. O modelo do protótipo (cerca de 1,7 MB) é gerado em `modelo/` e versionado na execução final, para o app abrir sem retreinar.
- Os testes usam dados **sintéticos** apenas para validar o código; nenhum resultado do projeto vem deles.
- A primeira execução costuma ser lenta no Windows (carregamento das bibliotecas); a seguinte é bem mais rápida.

## 9. Resultados
Os números abaixo vêm da execução com o dataset real (`results/`). Os tempos variam entre execuções da mesma máquina.

### 9.1 E1: comparação dos métodos (mesmo orçamento de 0,2% de alertas)
| Método | PR-AUC | ROC-AUC | Precisão | Recall | F1 | Acurácia | Tempo treino (s) | Tempo predição (s) |
|---|---|---|---|---|---|---|---|---|
| Isolation Forest | **0,1335** | 0,9447 | 0,2353 | 0,2817 | 0,2564 | 0,9973 | 0,34 | 0,37 |
| Z-score | 0,0780 | 0,9321 | 0,1294 | 0,1549 | 0,1410 | 0,9969 | 0,14 | 0,04 |
| LOF (subamostra de 20 mil) | 0,0331 | 0,8800 | 0,0706 | 0,0845 | 0,0769 | 0,9966 | 0,70 | 1,17 |
| Z-score com \|z\| ≥ 3 fixo (referência) | 0,0780 | 0,9321 | 0,0107 | 0,8732 | 0,0212 | 0,8656 | 0,14 | 0,04 |
| Sempre "normal" (referência) | 0,0017 | 0,5000 | 0 | 0 | 0 | 0,9983 | - | - |

Dos 170 alertas do Isolation Forest, cerca de 40 são fraudes (precisão de 23,5%, aproximadamente 140 vezes a taxa-base de 0,17%) e 40 das 142 fraudes do teste são encontradas (recall de 28%). O ranking é útil para priorizar, mas a PR-AUC de 0,13 é **modesta**: cerca de 72% das fraudes ficam fora dos alertas. A regra fixa \|z\| ≥ 3 acha 87% das fraudes, mas gera 11.546 alertas (13,6% das transações) com precisão de 1%.

### 9.2 E4: estabilidade com 10 sementes
PR-AUC média **0,108**, desvio-padrão **0,025**, mínimo 0,043 (semente 8) e máximo 0,136 (semente 6). O valor de 0,1335 do E1 (semente 42) está perto do topo dessa faixa e deve ser lido como a execução principal, e não como valor típico.

### 9.3 E2: parâmetros e tamanho do treino (um fator por vez)
- `n_estimators` (25 a 400): o tempo de treino cresce quase linearmente; a PR-AUC fica entre 0,10 e 0,13 sem tendência.
- `max_samples` (64 a 2.048): a PR-AUC sobe de 0,071 a 0,176 e o tempo também; o padrão (256) não é o melhor desta divisão. Não escolhemos o parâmetro olhando o teste, e esta análise é de sensibilidade.
- Registros de treino (19,9 mil a 198,6 mil, com `max_samples=256`): a PR-AUC fica entre 0,12 e 0,14 e o tempo cresce muito menos que proporcionalmente.

### 9.4 E3: limiar (`contamination`)
| contamination | Alertas | Precisão | Recall | F1 | Acurácia | PR-AUC |
|---|---|---|---|---|---|---|
| 0,0005 | 41 | 0,293 | 0,085 | 0,131 | 0,9981 | 0,1335 |
| 0,0010 | 84 | 0,298 | 0,176 | 0,221 | 0,9979 | 0,1335 |
| 0,0020 | 165 | 0,242 | 0,282 | 0,261 | 0,9973 | 0,1335 |
| 0,0050 | 443 | 0,140 | 0,437 | 0,212 | 0,9946 | 0,1335 |
| 0,0100 | 916 | 0,088 | 0,570 | 0,153 | 0,9895 | 0,1335 |
| 0,0200 | 1.738 | 0,058 | 0,704 | 0,106 | 0,9803 | 0,1335 |
| 0,0500 | 4.278 | 0,027 | 0,803 | 0,052 | 0,9508 | 0,1335 |

O ranking não muda (PR-AUC constante); só o equilíbrio entre falsos alarmes e fraudes perdidas. Em nenhuma linha a acurácia supera a do modelo que nunca aponta fraude (0,9983), o que mostra por que ela não serve como métrica principal.

### 9.5 Análise de erros (interpretação preliminar; o grupo deve revisar)
| Tipo | Índice | Posição no ranking (de 85.118) | Amount | Observação |
|---|---|---|---|---|
| Falso positivo | 274771 | 1 | 25.691,16 | variáveis extremas (V7 ≈ 100 desvios) |
| Falso positivo | 58465 | 2 | 19.656,53 | V20, V2, V23 muito altos |
| Falso positivo | 54018 | 3 | 11.898,09 | V20, V2, V23 muito altos |
| Falso negativo | 68633 | 70.468 | 1,18 | nenhuma variável acima de 1,6 desvio |
| Falso negativo | 56703 | 59.516 | 0,76 | variáveis dentro de ±2,3 desvios |
| Falso negativo | 50537 | 56.833 | 1,00 | variáveis dentro de ±1,4 desvio |
| Fraude que o LOF ranqueia melhor | 52521 | 53.692 | 105,99 | nenhuma variável isolada acima de 1,6 desvio |
| Fraude que o LOF ranqueia melhor | 72757 | 30.703 | 1,79 | V27 em 3,1 desvios |
| Fraude que o LOF ranqueia melhor | 112840 | 26.235 | 45,03 | variáveis discretas |

- **Falsos positivos:** são as maiores transações do conjunto, com variáveis em dezenas de desvios. O Isolation Forest isola o que é extremo, e extremo não é fraude: o método mede raridade.
- **Falsos negativos:** têm valor muito baixo e variáveis dentro de poucos desvios, ou seja, parecem normais em 30 dimensões e ficam na metade menos suspeita do ranking. *Hipótese ainda não verificada:* fraudes de valor mínimo podem ser testes de cartão.
- **Casos que o LOF ranqueia melhor:** *hipótese:* são incomuns pela **combinação** das variáveis (densidade local), e não por uma variável isolada.

[PREENCHER: interpretação final do grupo, caso a caso, em `docs/analise_de_erros.md`.]

## 10. Protótipo
`app.py` abre a **Central de Impostores** com duas abas: *Analisar transação* (sortear, editar o `Amount`, analisar, ver score, posição no teste, gráfico e variáveis mais extremas) e *Ranking de risco* (lote de demonstração ordenado por score, com inspeção de qualquer caso). O alerta é uma triagem experimental: não confirma fraude. O gráfico 2D é uma projeção (PCA) que mostra a variância explicada na legenda.

## 11. Limitações
- O Isolation Forest encontra o que é **raro**, e nem todo raro é fraude: compras legítimas de valor alto viram falsos positivos.
- Fraudes parecidas com transações normais passam despercebidas (cerca de 72% das fraudes do teste ficam fora dos alertas).
- **Teste pequeno e uma única divisão:** são 142 fraudes e uma divisão com uma semente. O E4 mede a variação do algoritmo, mas não a da divisão. Diferenças de 0,01 a 0,03 de PR-AUC não são conclusivas.
- É preciso escolher um limiar (`contamination`); o resultado operacional depende dele.
- O LOF usa cerca de 10% do treino; a comparação não vale para o LOF com todos os dados.
- `V1` a `V28` vêm de um PCA feito pelos autores do dataset **antes** da nossa divisão treino/teste (vazamento leve fora do nosso controle) e não são interpretáveis.
- A média e o desvio-padrão do Z-score e da padronização são sensíveis a valores extremos (como o `Amount` máximo de 25.691); o efeito de usar medidas robustas é investigado no Extra 2.
- O dataset é histórico (2013, dois dias) e o protótipo não usa dados reais.
- O gráfico 2D do protótipo explica pouca variância.
- Os tempos de execução variam entre rodadas e não devem ser lidos com precisão.

## 12. Integrantes e responsabilidades
| Integrante | GitHub |
|---|---|---|---|
| Felipe Hinke Betoni | felipebetoni |
| Gabriel Brune | GabrielBrune | 
| Ian Torres | itorresxp | 
| Matheus Borges | TheusBorges | 
| Pedro Sabbag | PedroSabbag | 

Funções propostas: dados/EDA; preparação; experimentos; protótipo; comunicação. Todos devem conseguir explicar o fluxo completo.

## 13. Próximos passos
1. **Checkpoint 1 (27/10):** repositório conectado, execução reproduzida por outro integrante e interpretação das descobertas e dos erros escrita.
2. **Checkpoint 2 (03/11):** rodar o Extra 2 (medidas robustas); repetir o E1 com várias divisões treino/teste para reduzir o efeito do teste pequeno; testar a hipótese dos valores mínimos (fração de fraudes com `Amount` < 2); concluir a análise de erros com a posição de cada caso também no LOF.
3. **Pré-banca (10/11):** congelamento técnico do protótipo.
4. **Entrega final:** 17/11 às 17h. Apresentação em 24/11.

## 14. Links
- Repositório: https://github.com/GabrielBrune/Isolation_Forest_Impostor
- Resumo expandido: `docs/resumo_expandido.md`
- Vídeo: [PREENCHER] • Aplicação: [PREENCHER] • Materiais (folder e carrossel): [PREENCHER]

## 15. Referências e Diário de IA
Referências em `docs/referencias.md`. O registro do uso de ferramentas de IA está em `docs/diario_ia.md`.

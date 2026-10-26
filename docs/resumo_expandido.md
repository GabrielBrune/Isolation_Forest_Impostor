# Detecção não supervisionada de anomalias em transações de cartão de crédito com Isolation Forest

**Autores:** | Integrante | GitHub |
| Felipe Hinke Betoni | felipebetoni
| Gabriel Brune | GabrielBrune 
| Ian Torres | itorresxp 
| Matheus Borges | TheusBorges 
| Pedro Sabbag | PedroSabbag 
**Instituição/curso:** Faculdades Integradas de Bauru(FIB) - Ciências da Computação 
**Disciplina:** Inteligência Artificial II • 2026

> Rascunho do Checkpoint 1. A seção 5 só pode ser preenchida com resultados gerados pelo código real. Reescrevam o texto com as palavras do grupo e confirmem as citações.

## 1. Introdução
A detecção de transações potencialmente fraudulentas é relevante para instituições financeiras, pois grandes volumes de operações precisam ser avaliados e a fraude é uma classe rara. Em cenários desbalanceados, a acurácia pode sugerir bom desempenho mesmo quando o sistema não identifica a classe de interesse; por isso, métricas de precisão-revocação são mais informativas (SAITO; REHMSMEIER, 2015).

Este projeto realiza uma investigação experimental com o conjunto público Credit Card Fraud Detection (KAGGLE/ULB; DAL POZZOLO et al., 2015). A técnica principal é o Isolation Forest (LIU; TING; ZHOU, 2008), que identifica observações que podem ser isoladas com poucas partições aleatórias. Ele é comparado a um baseline estatístico Z-score e ao Local Outlier Factor (LOF), que compara a densidade local de uma observação com a de seus vizinhos (BREUNIG et al., 2000). A comparação permite avaliar se um detector de anomalias traz ganho prático sobre alternativas mais simples.

## 2. Objetivos
**Objetivo geral.** Comparar o desempenho de Isolation Forest, Z-score e LOF na priorização de transações fraudulentas, com avaliação reproduzível e sem usar os rótulos de fraude no treino.

**Objetivos específicos.**
1. Caracterizar o dataset e verificar qualidade, duplicidades, valores ausentes e desbalanceamento.
2. Preparar os dados sem que informação do teste influencie o treino.
3. Construir o baseline Z-score e treinar Isolation Forest e LOF.
4. Comparar PR-AUC, precisão, recall, F1, acurácia e tempos de treino e de predição.
5. Investigar falsos positivos e falsos negativos e testar a sensibilidade a parâmetros e ao limiar.

**Pergunta de investigação.** Sem usar rótulos no treino, o Isolation Forest apresenta PR-AUC superior à do Z-score e à do LOF no mesmo split, e como o limiar de decisão equilibra falsos alarmes e fraudes não detectadas?

## 3. Relevância do estudo
Uma ferramenta de triagem pode ajudar analistas a priorizar transações para revisão, mas um alerta de anomalia não confirma fraude. A relevância está em avaliar os métodos de forma transparente, reconhecer limitações e considerar o custo de falsos positivos e negativos. O projeto também mostra por que métricas adequadas a classes raras são necessárias.

## 4. Materiais e métodos preliminares
Será usado o dataset Credit Card Fraud Detection, do Kaggle, com cerca de 284 mil transações e as variáveis `Time`, `Amount`, `V1` a `V28` (componentes de PCA anonimizadas) e `Class` (1 = fraude). Os números exatos serão confirmados no arquivo baixado.

A análise exploratória contará registros e colunas, medirá valores ausentes e duplicados, examinará o desbalanceamento, a distribuição de `Amount`, as variáveis que mais diferem entre as classes e os outliers pela regra do IQR. As linhas duplicadas serão removidas para que linhas idênticas não caiam em treino e teste ao mesmo tempo; a quantidade removida será reportada.

Os dados serão divididos em treino (70%) e teste (30%) com estratificação e semente fixa (42). A imputação e a padronização serão ajustadas apenas no treino, e `Class` será usada só na avaliação. O baseline Z-score usa o maior desvio absoluto padronizado por transação. Isolation Forest e LOF (modo `novelty`, 20 vizinhos, subamostra de 20 mil linhas do treino por custo computacional) serão treinados sem rótulos.

A métrica principal será a Average Precision (PR-AUC). Precisão, recall e F1 serão calculados com o **mesmo orçamento de alertas** (0,2% do teste) para os três métodos, além de acurácia (apenas como contraste), número de alertas e tempos de treino e predição. Serão feitos experimentos controlados: (E1) comparação dos detectores; (E2) `n_estimators`, `max_samples` e tamanho do treino, um fator por vez; (E3) limiar (`contamination`). Serão analisados ao menos três erros ou casos inesperados. O relatório informará parâmetros, semente e versões das bibliotecas (PEDREGOSA et al., 2011).

## 5. Resultados preliminares
**Pendente de execução com o dataset real.** Esta seção deve apresentar: (a) dimensões, duplicatas removidas e proporção de fraudes observadas; (b) três descobertas de EDA com gráfico ou tabela; (c) a tabela comparando os três métodos; (d) ao menos três exemplos de falsos positivos, falsos negativos ou casos inesperados. Copiar os valores de `results/` e interpretá-los; não inserir valores estimados.

## 6. Próximas etapas
Para o Checkpoint 2 (03/11): comparar os métodos, investigar os erros, avaliar `n_estimators`, `max_samples` e o limiar, e testar o efeito da variável `Time`. O protótipo web apresentará score, ranking de risco e inspeção de casos, deixando explícito que é triagem experimental.

## Referências
Ver `docs/referencias.md` e padronizar conforme as normas da disciplina.

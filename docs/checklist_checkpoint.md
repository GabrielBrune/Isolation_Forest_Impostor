# Checklist de autoverificação: Checkpoint 1 (27/10)
**Projeto:** Central de Impostores. Marque `[x]` somente depois de **verificar a evidência real** (arquivo, execução ou conversa com o grupo).
Itens marcados abaixo têm evidência das execuções feitas até aqui. As observações entre parênteses dizem o que ainda falta em itens ainda abertos.

## Entregas
- [x] Link do GitHub acessível ao professor e README atualizado: https://github.com/GabrielBrune/Isolation_Forest_Impostor 
- [x] Código/notebooks e instruções do dataset disponíveis (só vale depois do push)
- [x] Rascunho do resumo expandido revisado, reescrito pelo grupo e com citações (`docs/resumo_expandido.md`).
- [x] Pelo menos 5 referências, incluindo artigos científicos, com metadados conferidos (`docs/referencias.md`).
- [x] Este checklist revisado pelo grupo.
- [x] Diário de IA preenchido com interações reais (`docs/diario_ia.md`).

## A. Problema e objetivo
- [x] Problema, usuário e pergunta de investigação escritos no README e no resumo.
- [x] Todos os integrantes explicam o problema **antes** de citar o algoritmo.
- [x] O objetivo não se reduz a "aplicar o Isolation Forest" (é comparar e investigar).

## B. Dataset
- [x] Fonte identificada (Kaggle/ULB).
- [x] Dataset baixado e origem verificada (página `mlg-ulb/creditcardfraud`, `creditcard.csv`, 150,83 MB). **Licença ainda não conferida** (copiar da página para o README).
- [x] Registros e colunas confirmados no arquivo: 284.807 linhas e 31 colunas (`results/eda_summary.json`).
- [x] Variável-alvo `Class` identificada (0 = normal, 1 = fraude; usada só na avaliação).
- [x] Ausentes e duplicadas verificadas na execução: 0 ausentes; 1.081 duplicadas removidas (19 eram fraudes); ficaram 283.726 linhas e 473 fraudes.
- [ ] Outliers investigados (`results/eda4_outliers.csv`) e decisões justificadas (a investigação foi feita; **Optamos por não remover outliers. O objetivo do projeto é justamente detectar transações atípicas, e a análise exploratória mostrou que o sinal de fraude está concentrado nos extremos: entre 80% e 87% das fraudes são outliers pela regra do IQR em V12, V14 e V17, contra 2% a 5% das transações normais. Removê-los eliminaria a maior parte da informação sobre fraude. Ao mesmo tempo, ser outlier não equivale a ser fraude: 11,2% das transações normais também são outliers em Amount. Por isso o projeto trata a anomalia como um score contínuo, avaliado em ranking (PR-AUC), e não como uma regra de corte. Não encontramos indício de que os valores extremos sejam erros de registro.**).
- [x] Desbalanceamento confirmado com contagens reais: fraude = 0,167% das transações.
- [x] Escala considerada no código (padronização ajustada só no treino).
- [x] Vazamento revisado: linhas idênticas entre treino e teste = 0; PCA prévio dos autores registrado como limitação no README.

## C. EDA
- [x] Script de EDA criado (`src/eda.py`).
- [x] Script executado com o dataset real.
- [ ] Três descobertas interpretadas com evidências reais (gráficos `results/eda*.png`): os resultados existem; falta **escrever a interpretação** no resumo com palavras do grupo.
- [ ] Cada gráfico tem pergunta/insight associado.

## D. Preparação
- [x] Split estratificado com semente fixa (70/30, semente 42).
- [x] Imputação e padronização ajustadas só com o treino.
- [ ] Justificativas de limpeza escritas no resumo.

## E. Baseline
- [x] Baseline Z-score implementado.
- [x] Baseline executado e métrica registrada (`results/e1_resultados.csv`).
- [ ] Escolha e limitações do baseline explicadas.

## F. Pipeline ponta a ponta
- [x] Scripts `src/eda.py` e `src/experimentos.py` criados e `pytest` passando (6 testes, em dados sintéticos).
- [x] Dependências instaladas e execução sem erros na máquina do grupo com o dataset real (Python 3.14.5; EDA e pipeline completo com `--e4`).
- [ ] Evidências de execução guardadas e versionadas em `results/` (incluindo `run_info.json`): os arquivos existem localmente; falta o commit/push.
- [ ] Execução repetida por **outro integrante** (reprodutibilidade).

## G. GitHub
- [ ] Repositório criado e acessível ao professor (criado; falta conectar, dar push e confirmar acesso).
- [x] `.gitignore` não versiona o CSV.
- [ ] Modelo (1,7 MB) liberado no `.gitignore` e versionado, e `requirements.txt` com versões fixas (decidido: versionar na execução final).
- [ ] Sem credenciais ou dados proibidos no repositório (conferir com `git status` antes do push).
- [ ] Participação do grupo visível no histórico (commits de pessoas diferentes).

## H. Fundamentação
- [x] Referências iniciais listadas.
- [ ] Metadados e DOIs conferidos um a um.
- [ ] Citações integradas ao resumo (o rascunho já tem; revisar).
- [ ] Cada um explica a contribuição de pelo menos 2 referências.

## I. Divisão de responsabilidades
- [ ] Nomes reais no README (sem "Pessoa A..E").
- [ ] Ninguém concentra todo o desenvolvimento.
- [ ] **Todos** explicam o fluxo completo e conseguem explicar o bloco de código que lhes cabe.

## J. Diário de IA
- [x] Diário iniciado com as ferramentas usadas até agora.
- [ ] Campos `[PREENCHER]` preenchidos com fatos verdadeiros.
- [ ] Validação das sugestões técnicas registrada.

## K. Resultados e interpretação (adicionado após a execução com o dataset real)
- [x] E1 a E3, E4 (10 sementes), ablação sem `Time` e análise de erros executados (IF, Z-score e LOF).
- [x] Resultado principal registrado: PR-AUC do Isolation Forest 0,1335 (semente 42) e 0,108 ± 0,025 entre 10 sementes; Z-score 0,078; LOF 0,033 (subamostra de 20 mil do treino).
- [ ] Tabela de resultados do README preenchida a partir de `results/` (reportar média ± desvio do E4 ao lado do 0,1335).
- [ ] Hipóteses atualizadas no README com o que os dados sustentam (H1 sustentada nesta divisão, com ressalvas; H2: `max_samples` pequeno refutada; tempo de treino cresce menos que proporcionalmente aos registros, sem afirmar valor exato).
- [ ] Interpretação dos 9 casos de erro escrita (fora do CSV, que é recriado a cada execução).
- [ ] Checagem da hipótese de valor mínimo (fração de fraudes com Amount < 2) feita.
- [ ] Coluna `posicao_no_ranking_LOF` adicionada em `analisar_erros` e pipeline reexecutado antes da versão final.
- [ ] Gráfico E2 regenerado com eixos fixos e números do eixo x sem sobreposição.

## Demonstração de 10 minutos
- [ ] 0–2 min: problema e pergunta de investigação.
- [ ] 2–4 min: dataset e 3 descobertas.
- [ ] 4–6 min: baseline e resultado real.
- [ ] 6–8 min: execução ponta a ponta.
- [ ] 8–10 min: próxima hipótese/experimento para 03/11.

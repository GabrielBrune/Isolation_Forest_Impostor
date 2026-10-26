# Diário de IA

Projeto: Central de Impostores / Isolation Forest (IA II, 2026). Registro de acompanhamento (não avaliativo).
Regra: registre só o que aconteceu de fato. 

## Entrada 1: entender o tema e preencher o canvas
- **Data:** 20/10/2026
- **Ferramenta e versão:** Claude (Anthropic), Claude Sonnet 5.5, pelo claude.ai
- **Prompt (resumo):** pedi uma explicação do Isolation Forest e da dica do professor ("Existe um impostor aqui?") e ajuda para preencher o canvas inicial usando o exemplo de Naive Bayes como modelo.
- **O que a IA respondeu (resumo):** explicou a ideia de isolamento por cortes aleatórios e sugeriu campos do canvas (tema, problema, dataset de fraude de cartão, baseline, hipóteses, experimentos, protótipo).

## Entrada 2: leitura do cronograma e do checklist
- **Data:** 20/10/2026
- **Ferramenta e versão:** Claude Sonnet 5.5, pelo claude.ai
- **Prompt (resumo):** enviei o PDF do cronograma e os documentos de modelos e checklists do professor e pedi um resumo do que entregar em cada data.
- **Como usei a resposta:** serviu de plano de trabalho até 27/10.
- **Como verifiquei:** Validei com o grupo se todos entendia como ok a saida gerada pelo modelo

## Entrada 3: geração do primeiro código (pipeline e protótipo)
- **Data:** 20/10/2026
- **Ferramenta e versão:** Claude Sonnet 5.5, pelo claude.ai
- **Prompt (resumo):** pedi o projeto em código para ver como ficaria e o que precisava ser feito.
- **O que a IA entregou:** `projeto_impostor.py` (EDA, baseline Z-score, Isolation Forest, experimentos E1 a E3, análise de erros) e `app.py` (protótipo em Streamlit).
- **Como usei:** executei os arquivos no meu computador com o `creditcard.csv` real e o código rodou até o final, gerando tabelas e gráficos. 
- **Como verifiquei:** a IA testou o código com dados sintéticos no mesmo formato do dataset, e eu executei com o dataset real. 
- **Observação:** esse primeiro código foi **substituído** pelo repositório unificado (Entrada 7); a execução de então não valida o código novo.

## Entrada 4: problemas de instalação
- **Data:** 20/10/2026
- **Ferramenta e versão:** Claude Sonnet 5.5, pelo claude.ai
- **Prompt (resumo):** enviei prints de erros no terminal (`pip` e `python` não reconhecidos, nome errado do `requirements.txt`).
- **Como usei a resposta:** usei `py -m pip install -r requirements.txt` e consegui instalar e rodar.
- **Como verifiquei:** o código rodou até a mensagem final.

## Entrada 5: checklist do Checkpoint 1, README e rascunho do resumo
- **Data:** 20/10/2026
- **Ferramenta e versão:** Claude Sonnet 5.5, pelo claude.ai
- **Prompt (resumo):** enviei o checklist do Checkpoint 1 e pedi tudo o que eu precisava fazer e entregar.
- **O que a IA entregou:** ajustes do código (duplicatas, outliers e checagem de vazamento), rascunho de README, rascunho do resumo expandido, lista de referências e um modelo de diário.
- **Como usei:** Segui o passo a passo que ela me passou
- **Como verifiquei:** Sem verificação.

## Entrada 6: organização do checkpoint com outra ferramenta
- **Data:** 20/10/2026
- **Ferramenta:** ChatGPT
- **Modelo/versão:** GPT-6 Instant
- **Objetivo/prompt (resumo):** organizar as entregas do Checkpoint 1 a partir das instruções da disciplina e do canvas.
- **O que a IA entregou:** estrutura de documentação, checklist, plano de execução, `src/eda.py` e `src/train.py` simples, rascunho de resumo e lista de referências.
- **Como a resposta foi utilizada:** documentação, checklist e referências foram aproveitados no repositório unificado; o código foi substituído (Entrada 7).
- **Validação:** Usando o Sonnet 5.5 para validar com base nos requisitos descritos nos pdfs entregue pelo professor

## Entrada 7: unificação dos dois pacotes em um repositório 
- **Data:** 20/10/2026
- **Ferramenta e versão:** Claude Sonnet 5.5, pelo claude.ai
- **Prompt (resumo):** enviei os dois pacotes gerados pelo grupo (um do Claude, outro do ChatGPT) e pedi uma revisão e a mescla em um repositório só. Depois pedi que todas as funções tivessem comentários explicativos.
- **O que a IA entregou:** repositório `central-impostores` (src/common.py, src/eda.py, src/experimentos.py, app.py, testes, README e docs). Mudanças em relação aos pacotes: mesmo orçamento de alertas para os três métodos, medição de tempo de treino e de predição, tamanho do treino no E2, ablação sem `Time`, remoção de duplicatas, protótipo com ranking de risco e inspeção de casos. A pedido, docstrings em todas as funções e comentários nos pontos-chave do fluxo.
- **Teste feito pela IA:** testes automatizados e execução completa **somente com dados sintéticos** (nenhum resultado real). A execução com o `creditcard.csv` verdadeiro foi feita depois pelo grupo (Entrada 10).
- **Como usei:** usei o repositório como base do projeto e rodei no meu computador (Windows, Python 3.14.5).
- **Como verifiquei:** `pytest` passou (6 testes); EDA e pipeline rodaram com o dataset real (Entrada 10). 

## Entrada 8: entender o tema, Canvas e versão para leigos 
- **Data:** 25/10/2026
- **Ferramenta e versão:** Claude Sonnet 5.5, pelo claude.ai
- **Prompt (resumo):** perguntei o que é o Isolation Forest, como funciona e onde se aplica; pedi a estrutura do Canvas em .docx e uma versão simplificada com analogia para leigos.
- **O que a IA entregou:** explicação (isolamento por cortes aleatórios, score de anomalia, aplicações, limitações), Canvas em .docx seguindo o exemplo de Naive Bayes do professor, e uma explicação para leigos com a analogia do jogo de perguntas para achar a pessoa diferente numa festa.
- **Como usei:** o que foi aproveitado, alterado e descartado
- **Como verifiquei:** conferi no artigo original de Liu, Ting e Zhou (2008) e na documentação do scikit-learn

## Entrada 9: revisão do Canvas do grupo, cronograma e checklist 
- **Data:** 25/10/2026
- **Ferramenta e versão:** Claude Sonnet 5.5, pelo claude.ai
- **Prompt (resumo):** enviei o Canvas desenvolvido pelo grupo, o Caderno dos 10 Desafios, o cronograma e o checklist do Checkpoint 1, e pedi uma análise do que estava certo e do que faltava.
- **O que a IA entregou:** lista de problemas (análise de erros marcada como opcional, E3 misturando fatores e entendimento do `contamination`, hipóteses pouco testáveis, falta de métricas, baseline Z-score vago, protótipo sem ranking de risco e inspeção de casos, definição de "normal" ausente); Canvas revisado com ficha de experimentos preenchida (sem resultados), planilha comparando o Canvas com o Caderno, plano de trabalho por checkpoint e rascunhos do marco de 13/10 (descrição das variáveis, fluxo, pergunta de investigação, referências candidatas).
- **Aviso:** as referências sugeridas vieram da memória da IA e podem ter metadados errados.
- **Como usei:** 
- **Como verifiquei:** 

## Entrada 10: execução com o dataset real e leitura dos resultados 
- **Data:** 25/10/2026
- **Ferramenta e versão:** Claude Sonnet 5.5, pelo claude.ai
- **Prompt (resumo):** colei no chat a saída da EDA e do pipeline (`--sem-lof` e depois `--e4`) e os gráficos de `results/` e pedi a interpretação.
- **O que a IA entregou:** conferência das contagens (284.807 linhas, 492 fraudes, 1.081 duplicadas, 283.726 linhas e 473 fraudes depois da limpeza), veredito por hipótese, alertas sobre o ruído (teste com 142 fraudes e uma única divisão), orientação de reportar a PR-AUC como média ± desvio das 10 sementes (0,108 ± 0,025; 0,1335 é a execução de semente 42), leitura dos gráficos e modelo de interpretação para os casos de erro.
- **Como usei:** rodei os comandos no meu computador e colei as saídas reais no chat.
- **Como verifiquei:** as saídas vieram da execução real. As contagens esperadas foram citadas de memória pela IA e bateram com a execução.

## Entrada 11: ambiente, execução no Windows, publicação e GitHub 
- **Data:** 25/10/2026
- **Ferramenta e versão:** Claude Sonnet 5.5, pelo claude.ai
- **Prompt (resumo):** pedi passo a passo para rodar o projeto no Windows e fiz perguntas sobre NPU, lentidão do primeiro `pytest`, o botão "Deploy" do Streamlit, onde baixar dados reais, versões do `requirements.txt`, o que vai no `.gitignore` e como conectar a pasta ao repositório do GitHub.
- **O que a IA entregou:** passo a passo (`py -m venv`, instalação, EDA, pipeline, protótipo e git); orientação de que o código roda na CPU; diagnóstico provável de execução a frio para o `pytest` de 116 s (a segunda execução levou 3,52 s); versões fixas copiadas do `pip freeze` para o `requirements.txt`; orientação de versionar o modelo (1,7 MB) só na execução final; marcação do checklist do Checkpoint 1 com o que já tem evidência real.
- **Como usei:** seguimos o passo a passo entrege pelo agente
- **Como verifiquei:** os comandos foram executados no meu computador e as saídas colei no chat.

## Entrada 12: ajustes de gráficos e de código sugeridos pela IA 
- **Data:** 25/10/2026
- **Ferramenta e versão:** Claude Sonnet 5.5, pelo claude.ai
- **O que a IA sugeriu:** fixar os eixos do gráfico E2 (`set_ylim`), inclinar os números do eixo x (`tick_params`), acrescentar a coluna `posicao_no_ranking_LOF` na análise de erros e conferir a hipótese de valor mínimo (Amount < 2).
- **Como usei:** o gráfico E2 foi regenerado com os eixos fixos.
- **Como verifiquei:** Comparei os dois resultados gerados.

## Registro de descartes ou correções
- A H2 do canvas dizia que o tempo de treino do Isolation Forest cresce com o número de registros. Isso estava incorreto (depende de `n_estimators` e `max_samples`); foi corrigido no README, no código e na ficha de experimentos.
- Depois da primeira execução real, a IA classificou a H2 como "confirmada" (tempo de treino quase plano). O gráfico regenerado mostrou o tempo subindo cerca de 2 vezes com 10 vezes mais registros, e a IA corrigiu: o tempo cresce **menos que proporcionalmente** aos registros, e as medições são ruidosas.
- A IA supôs que o notebook tinha 8 GB de RAM; ele tem 16 GB. A hipótese sobre memória foi descartada.
- A IA previu que o Z-score poderia ganhar do Isolation Forest por causa das variáveis concentradas (V14, V17); nesta divisão, o Isolation Forest teve PR-AUC maior. A previsão não se confirmou.
- O primeiro trecho de código que a IA passou para a coluna `posicao_no_ranking_LOF` daria erro nos falsos positivos e negativos; foi substituído por uma versão corrigida (`t.loc[indice, "posicao_lof"]`).

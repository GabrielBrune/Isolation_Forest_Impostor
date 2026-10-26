# Referências

Pontos de partida. Antes da entrega, **abram cada link/DOI**, confirmem os metadados (autores, ano, título, veículo) e apliquem o padrão de citação exigido pela disciplina.

1. LIU, Fei Tony; TING, Kai Ming; ZHOU, Zhi-Hua. **Isolation Forest.** In: *2008 Eighth IEEE International Conference on Data Mining*. IEEE, 2008. DOI: https://doi.org/10.1109/ICDM.2008.17
   **Contribuição:** artigo original do Isolation Forest, método principal do projeto.

2. BREUNIG, Markus M.; KRIEGEL, Hans-Peter; NG, Raymond T.; SANDER, Jörg. **LOF: Identifying Density-Based Local Outliers.** In: *Proceedings of the 2000 ACM SIGMOD International Conference on Management of Data*. ACM, 2000. DOI: https://doi.org/10.1145/342009.335388
   **Contribuição:** apresenta o Local Outlier Factor, usado como método comparativo.

3. SAITO, Takaya; REHMSMEIER, Marc. **The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets.** *PLOS ONE*, v. 10, n. 3, e0118432, 2015. DOI: https://doi.org/10.1371/journal.pone.0118432
   **Contribuição:** fundamenta o uso de precisão-recall (PR-AUC) em dados desbalanceados.

4. DAL POZZOLO, Andrea; CAELEN, Olivier; JOHNSON, Reid A.; BONTEMPI, Gianluca. **Calibrating Probability with Undersampling for Unbalanced Classification.** In: *2015 IEEE Symposium Series on Computational Intelligence*. IEEE, 2015. DOI: https://doi.org/10.1109/SSCI.2015.33
   **Contribuição:** trabalho associado ao dataset de fraude de cartão e à discussão de classes desbalanceadas. É sobre classificação supervisionada; usar como referência complementar, deixando claro que o projeto é de detecção não supervisionada.

5. KAGGLE / ULB. **Credit Card Fraud Detection** [conjunto de dados]. https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
   **Contribuição:** fonte dos dados. Confirmar autoria, licença e metadados na página vigente.

6. PEDREGOSA, Fabian et al. **Scikit-learn: Machine Learning in Python.** *Journal of Machine Learning Research*, v. 12, p. 2825–2830, 2011. https://www.jmlr.org/papers/v12/pedregosa11a.html
   **Contribuição:** biblioteca usada para implementar e reproduzir os modelos.

7. Documentação oficial do scikit-learn: [IsolationForest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html), [LocalOutlierFactor](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html) e [métricas de avaliação](https://scikit-learn.org/stable/modules/model_evaluation.html).
   **Contribuição:** parâmetros e interpretação das funções; complementa os artigos, não os substitui.

## Candidatas para limitações e discussão (verificar metadados antes de citar)
8. LIU, Fei Tony; TING, Kai Ming; ZHOU, Zhi-Hua. **Isolation-Based Anomaly Detection.** *ACM Transactions on Knowledge Discovery from Data*, 2012.
9. HARIRI, Sahand; KIND, Matias Carrasco; BRUNNER, Robert J. **Extended Isolation Forest.** *IEEE Transactions on Knowledge and Data Engineering*. (cortes paralelos aos eixos geram artefatos nos scores; discute a variante estendida)

## Onde cada fonte entra no texto
- Introdução e métricas: Saito e Rehmsmeier (2015).
- Método principal: Liu, Ting e Zhou (2008).
- Método comparativo: Breunig et al. (2000).
- Dados: página do Kaggle e Dal Pozzolo et al. (2015).
- Implementação e reprodutibilidade: Pedregosa et al. (2011) e documentação oficial.

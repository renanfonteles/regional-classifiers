# Pattern Classification Based on Regional Models

Code, data and models for manuscript "Pattern Classification Based on Regional Models" 

### TODO

 - [ ] Fig 8, 9, 10 (Main-paper-results part 1, 2 e 3) - TODO: Gerar tabelas que concidam com o artigo
 - [ ] Definir orientações para geração de resultados com diferentes configurações (datasets, modelos, etc)
 - [ ] Definir notebook referente as figuras 5 e 6
 - [ ] Comentar código e notebooks

### Dependencies

- numpy
- scikit
- tensorflow
- plotly (for visualization)

### Datasets

- Parkinson [https://archive.ics.uci.edu/ml/datasets/parkinsons]
- Vertebral Column [http://archive.ics.uci.edu/ml/datasets/vertebral+column]
- Wall-following [https://archive.ics.uci.edu/ml/datasets/Wall-Following+Robot+Navigation+Data]

<hr>

### Paper results

#### (Section 4.3) Part 1: global vs local-regional linear classification model

| Subsection                                                               | Output                                            |
|--------------------------------------------------------------------------|---------------------------------------------------|
| (Subsection 4.3.1) **Experiment I:** global vs regional LSC-LBF              | <ul><li>Table 3</li><li>Figures 5 and 6</li></ul> |
| (Subsection 4.3.2) **Experiment II:** influence of the number of local regions <br /> in classification performance  | <ul><li>Figure 7</li></ul> |

#### (Section 4.4) Part 2: global vs local-regional non-linear classification model

| Subsection                                                                                                               | Output                                   |
|------------------------------------------------------------------------------------------------------------------------|------------------------------------------|
| (Subsection 4.4.4) **Experiment III:** global vs local vs regional LSSVM                                                   | <ul><li>Tables 5, 6 and 7</li><li> Figures 8, 9 and 10</li></ul> |
| (Subsection 4.4.5) **Experiment IV:** optimal selection of number of local regions <br />based on ensemble of clustering indices | <ul><li>Figure 10</li></ul>                              |

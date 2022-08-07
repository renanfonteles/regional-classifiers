# Pattern Classification Based on Regional Models

Code, data and models for manuscript "Pattern Classification Based on Regional Models" 

### TODO
 - [ ] Gerar tabelas referente ao Experimento III (Notebook Main3)
 - [ ] Corrigir Extra2-Preliminary results [G-LSSVM and L-LSSVM].ipynb
 - [ ] Inserir código e resultados referente aos Experimentos I e II (Notebooks Main1 e Main2)
 - [ ] Definir quais notebooks "Extras" ou "Scripts" vão ser incluídos ao "Main"
 - [ ] Definir orientações para geração de resultados com diferentes configurações (datasets, modelos, etc)
 - [ ] Comentar código e notebooks
 
 **Obs:** Notebooks iniciados como "Main" será incluído ao Code Ocean (CO). Notebooks iniciados com "Extra" e "Script" precisam ser avaliados para ver se vai para o "Main" ou se servirão como apêndices.

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
| (Subsection 4.3.1) **Experiment I:** global vs regional LSC-LBF <br/> [*Main-paper-results-note4.ipynb*](https://github.com/renanfonteles/regional-classifiers/blob/main/Main-paper-results-part1.ipynb)              | <ul><li>Table 3</li><li>Figures 5 and 6</li></ul> |
| (Subsection 4.3.2) **Experiment II:** influence of the number of local regions <br /> in classification performance <br/> [*Main-paper-results-note4.ipynb*](https://github.com/renanfonteles/regional-classifiers/blob/main/Main-paper-results-part1.ipynb)  | <ul><li>Figure 7</li></ul> |

#### (Section 4.4) Part 2: global vs local-regional non-linear classification model

| Subsection                                                                                                               | Output                                   |
|------------------------------------------------------------------------------------------------------------------------|------------------------------------------|
| (Subsection 4.4.4) **Experiment III:** global vs local vs regional LSSVM <br/> [*Main-paper-results-note4.ipynb*](https://github.com/renanfonteles/regional-classifiers/blob/main/Main-paper-results-part1.ipynb)                                                   | <ul><li>Tables 5, 6 and 7</li><li> Figures 8 and 9</li></ul> |
| (Subsection 4.4.5) **Experiment IV:** optimal selection of number of local regions <br />based on ensemble of clustering indices <br/> [*Main-paper-results-note4.ipynb*](https://github.com/renanfonteles/regional-classifiers/blob/main/Main-paper-results-part1.ipynb)</span> | <ul><li>Figure 10</li></ul>                              |

# Pattern Classification Based on Regional Models

Code, data and models for manuscript "Pattern Classification Based on Regional Models" submitted to Applied Soft Computing.

### Abstract
In a supervised setting, the global classification paradigm leverages the whole training
data to produce a single class discriminative model. Alternatively, the local
classification approach builds multiple base classifiers, each of them using a small
subset of the training data. In this paper, we take a path to stand in-between the global
and local approaches. We introduce a two-level clustering-based method in which
base classifiers operate on a larger portion of the input space than in the traditional
local paradigm. In particular, we first obtain a grained input representation by
employing a Self-Organizing Map (SOM) to the inputs. We then apply a clustering
algorithm (e.g., K-Means) to the SOM units to define input regions a subset of input
samples associated with a specific cluster of SOM units. We refer to this approach as
regional classification. We demonstrate the effectiveness of regional classification on
several benchmarks. Also, we study the impact of 1) adopting linear and nonlinear
base classifiers (e.g., least-square support vector machines) and 2) using cluster
validation indexes to determine the optimal number of clusters. Based on the
experiments, the regional classification approach achieves competitive performance
compared to its global and local counterparts, especially when equipped with linear
base classifiers

### Citation
Please use the following bibtex entry:
```
@article{DRUMOND2022109592,
title = {Pattern classification based on regional models},
journal = {Applied Soft Computing},
pages = {109592},
year = {2022},
issn = {1568-4946},
doi = {https://doi.org/10.1016/j.asoc.2022.109592},
url = {https://www.sciencedirect.com/science/article/pii/S156849462200641X},
author = {Rômulo B.P. Drumond and Renan F. Albuquerque and Guilherme A. Barreto and Amauri H. Souza},
keywords = {Pattern classification, Local models, Regional models, Self-Organizing Maps, Least squares support vector machine},
abstract = {In a supervised setting, the global classification paradigm leverages the whole training data to produce a single class discriminative model. Alternatively, the local classification approach builds multiple base classifiers, each of them using a small subset of the training data. In this paper, we take a path to stand in-between the global and local approaches. We introduce a two-level clustering-based method in which base classifiers operate on a larger portion of the input space than in the traditional local paradigm. In particular, we first obtain a grained input representation by employing a Self-Organizing Map (SOM) to the inputs. We then apply a clustering algorithm (e.g., K-Means) to the SOM units to define input regions — a subset of input samples associated with a specific cluster of SOM units. We refer to this approach as regional classification. We demonstrate the effectiveness of regional classification on several benchmarks. Also, we study the impact of (1) adopting linear and nonlinear base classifiers (e.g., least-square support vector machines) and (2) using cluster validation indexes to determine the optimal number of clusters. Based on the experiments, the regional classification approach achieves competitive performance compared to its global and local counterparts, especially when equipped with linear base classifiers.}
}
```

### Code ocean
The code and data required to reproduce the experiments and findings of this paper are available in code ocean platform 
by the following link: https://codeocean.com/capsule/9667739/tree.

### Dependencies

- pickel
- pandas
- numpy
- scikit
- matplotlib
- seaborn
- plotly

### Datasets

- Parkinson [https://archive.ics.uci.edu/ml/datasets/parkinsons]
- Vertebral Column [http://archive.ics.uci.edu/ml/datasets/vertebral+column]
- Wall-following [https://archive.ics.uci.edu/ml/datasets/Wall-Following+Robot+Navigation+Data]

<hr>

### Main paper results

##### (Section 4.3) Part 1: global vs local-regional linear classification model
 - (Subsection 4.3.2) **Experiment II:** influence of the number of local regions in classification performance 
   - *Results-Part1 (Experiment II).ipynb*

##### (Section 4.4) Part 2: global vs local-regional non-linear classification model
- (Subsection 4.4.4) **Experiment III:** global vs local vs regional LSSVM <br/> 
  - *Results-Part2 (Experiment III).ipynb*


- (Subsection 4.4.5) **Experiment IV:** optimal selection of number of local regions based on ensemble of clustering indices
  - *Results-Part2 (Experiment IV).ipynb*

##### Extra contents
- 2D and 3D visualization of regional modeling process <br/> 
  - *Main-paper-extras-regional-visualization.ipynb*

- Homogeneous and empty region analysis <br/> 
  - *Main-paper-extras-homogeneous-analysis.ipynb*

#### Additional notes

<ul>
    <li>The hyperparameters regarding LSSVM, as well as the number of local/regional regions are pretrained and loaded previoustly to reduce computational cost.</li>
    <li>The random states for Monte Carlo simulation are also predefined.</li>
</ul>

<!---
| (Subsection 4.3.1) **Experiment I:** global vs regional LSC-LBF <br/>  *<ul><li>Results-Part2 (Experiment IV).ipynb</li></ul>*             | <ul><li>Table 3</li><li>Figures 5 and 6</li></ul> |

-->
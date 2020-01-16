# NeuralBandit

This WIP repository reproduces the experiments of the [NeuralBandit](https://hal.archives-ouvertes.fr/hal-01117311/document) and [BanditForest](http://proceedings.mlr.press/v51/feraud16.html) papers. The code of BanditForest is available [here](https://www.researchgate.net/publication/308305599_Test_code_for_Bandit_Forest_algorithm).

### Table of contents
- [Installation](#installation)
- [Steps to reproduce](#steps-to-reproduce)
- [Citations](#citations)
    + [NeuralBandit](#neuralbandit-1)
    + [BanditForest](#banditforest)

### TODO List
* Automating the run of experiments
* NeuralBandit Paper
    * Implementation of NeuralBandit.A and .B
    * Adding concept drift to the Game

* BanditForest Paper
    * Adding Census et Adult dataset
    * Adding Noise to the reward

### Installation
Use ```pip install neuralbandit``` to install the package from pip or clone the repository and install the package from the sources with the package manager [poetry](https://python-poetry.org/) by using the command ```poetry install```.

### Steps to reproduce
```python

from neuralbandit import get_cov_dataset, ContextualBanditGame, NeuralBandit
from neuralbandit.sota import RandomBandit, BanditTron, LinUCB

dataset = get_cov_dataset()

cumulative_reward = 0
T = int(2.5E6)

game = ContextualBanditGame(dataset, T)
#player = RandomBandit(dataset.K)
#player = LinUCB(dataset.K, dataset.D)
player = NeuralBandit(dataset.K, dataset.D, layer_count = 1, layer_size = 64, gamma = 0.05)

for t in tqdm(range(T)):
        context = game.get_context()
        action = player.select(context)
        reward = game.play(action)
        
        cumulative_reward += reward
        
        player.observe(action, context, reward)

cumulative_regret = (T*game.optimal_accuracy - cumulative_reward)
print(cumulative_regret)
```

### Citations
#### NeuralBandit
```
@InProceedings{10.1007/978-3-319-12637-1_47,
author="Allesiardo, Robin
and F{\'e}raud, Rapha{\"e}l
and Bouneffouf, Djallel",
editor="Loo, Chu Kiong
and Yap, Keem Siah
and Wong, Kok Wai
and Teoh, Andrew
and Huang, Kaizhu",
title="A Neural Networks Committee for the Contextual Bandit Problem",
booktitle="Neural Information Processing",
year="2014",
publisher="Springer International Publishing",
address="Cham",
pages="374--381",
abstract="This paper presents a new contextual bandit algorithm, NeuralBandit, which does not need hypothesis on stationarity of contexts and rewards. Several neural networks are trained to modelize the value of rewards knowing the context. Two variants, based on multi-experts approach, are proposed to choose online the parameters of multi-layer perceptrons. The proposed algorithms are successfully tested on a large dataset with and without stationarity of rewards.",
isbn="978-3-319-12637-1"
}
```
#### Selection of Learning Expert
The paper **Selection of Learning Expert** provides theoretical insights on the methodology used by the commitee of neural networks of the **NeuralBandit** paper.
```
@INPROCEEDINGS{7965962,
author={R. {Allesiardo} and R. {Feraud}},
booktitle={2017 International Joint Conference on Neural Networks (IJCNN)},
title={Selection of learning experts},
year={2017},
volume={},
number={},
pages={1005-1010},
keywords={game theory;learning (artificial intelligence);pattern classification;learning expert selection;online classification models;expert parametrization;hypothesis space;contextual bandits;adversarial problem;successive elimination algorithm;EXP 3 algorithm;bandit forests;Context;Approximation algorithms;Approximation error;Context modeling;Estimation error;Complexity theory;Stochastic processes},
doi={10.1109/IJCNN.2017.7965962},
ISSN={2161-4407},
month={May},}
```
#### BanditForest
```
@InProceedings{pmlr-v51-feraud16,
  title = 	 {Random Forest for the Contextual Bandit Problem},
  author = 	 {Raphaël Féraud and Robin Allesiardo and Tanguy Urvoy and Fabrice Clérot},
  booktitle = 	 {Proceedings of the 19th International Conference on Artificial Intelligence and Statistics},
  pages = 	 {93--101},
  year = 	 {2016},
  editor = 	 {Arthur Gretton and Christian C. Robert},
  volume = 	 {51},
  series = 	 {Proceedings of Machine Learning Research},
  address = 	 {Cadiz, Spain},
  month = 	 {09--11 May},
  publisher = 	 {PMLR},
  pdf = 	 {http://proceedings.mlr.press/v51/feraud16.pdf},
  url = 	 {http://proceedings.mlr.press/v51/feraud16.html},
  abstract = 	 {To address the contextual bandit problem, we propose an online random forest algorithm. The analysis of the proposed algorithm is based on the sample complexity needed to find the optimal decision stump. Then, the decision stumps are recursively stacked in a random collection of decision trees, BANDIT FOREST. We show that the proposed algorithm is optimal up to logarithmic factors. The dependence of the sample complexity upon the number of contextual variables is logarithmic. The computational cost of the proposed algorithm with respect to the time horizon is linear. These analytical results allow the proposed algorithm to be efficient in real applications , where the number of events to process is huge, and where we expect that some contextual variables, chosen from a large set, have potentially non-linear dependencies with the rewards. In the experiments done to illustrate the theoretical analysis, BANDIT FOREST obtain promising results in comparison with state-of-the-art algorithms.}
}

```

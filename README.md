# NeuralBandit

This WIP repository reproduces the experiments of the [NeuralBandit](https://hal.archives-ouvertes.fr/hal-01117311/document) and [BanditForest](http://proceedings.mlr.press/v51/feraud16.html) papers. The code of BanditForest is available [here](https://www.researchgate.net/publication/308305599_Test_code_for_Bandit_Forest_algorithm).

### TODO List
#### NeuralBandit Paper
* Implementation of NeuralBandit1 (it could be useful!)
* Implementation of NeuralBandit2
* Adding concept drift to the Game

#### BanditForest Paper
* Adding Census et Adult dataset
* Adding Noise to the reward

### Steps to reproduce
```python

from neuralbandit import get_cov_dataset, ContextualBanditGame
from neuralbandit.algorithm import RandomBandit, BanditTron, LinUCB

dataset = get_cov_dataset()

cumulative_reward = 0
T = int(2.5E6)

game = ContextualBanditGame(dataset, T)
#player = RandomBandit(dataset.K)
#player = BanditTron(dataset.K, dataset.D)
player = LinUCB(dataset.K, dataset.D)

for t in range(T):
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

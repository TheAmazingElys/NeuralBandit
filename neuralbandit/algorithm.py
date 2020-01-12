import torch, numpy as np
from random import randrange

class RandomBandit():
    """
    The Random Bandit
    """
    def __init__(self, K):
        self._K = K
        
    def select(self, x = None):
        return randrange(self._K)
    
    def observe(self, k, x, y):
        """
        Do nothing, is here just for compatibility of the API
        """
        pass


class BanditTron():
    """
    Implementation of the BanditTron paper (https://www.cse.huji.ac.il/~shais/papers/TewariShKa08.pdf)
    """
    def __init__(self, K, D, gamma = 0.05):
        
        self._K = K
        self._D = D
        self._gamma = gamma
        
        self._weights = torch.zeros(D,K)

    def select(self, contexts):
        with torch.no_grad():
            XW =  contexts.mm(self._weights)
            _, self.best_arms = torch.max(XW, 1)

            mask = torch.zeros(XW.shape).scatter (1, self.best_arms.unsqueeze(1), 1.0)
            self._probabilities = (1 - self._gamma) * mask + torch.zeros((len(XW), self._K)) + self._gamma/self._K

            k = [np.random.choice(self._K, p=p.data.numpy()/sum(p.data.numpy())) for p in self._probabilities]

            return k[0]
    
    def observe(self, k, context, reward):
        """
        Don't support batches of rewards
        """
        with torch.no_grad():
            context = context
            reward = reward

            self.update = context * ((reward - 1)/self._probabilities[0][k] - 1)

            self._weights[:,k:k+1] += self.update.squeeze(0).unsqueeze(1)
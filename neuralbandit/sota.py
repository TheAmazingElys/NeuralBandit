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
        Does nothing, is here just for compatibility of the API
        """
        pass


class BanditTron():
    """
    Implementation of the BanditTron paper (https://www.cse.huji.ac.il/~shais/papers/TewariShKa08.pdf)
    """
    def __init__(self, action_count, context_dimension, gamma = 0.05):
        
        self._K = action_count
        self._D = context_dimension
        self._gamma = gamma
        
        self._weights = torch.zeros(context_dimension, action_count)

    def select(self, context):
        with torch.no_grad():
            XW =  context.mm(self._weights)
            _, self.best_actions = torch.max(XW, 1)

            mask = torch.zeros(XW.shape).scatter (1, self.best_actions.unsqueeze(1), 1.0)
            self._probabilities = (1 - self._gamma) * mask + torch.zeros((len(XW), self._K)) + self._gamma/self._K

            k = [np.random.choice(self._K, p=p.data.numpy()/sum(p.data.numpy())) for p in self._probabilities]

            return k[0]
    
    def observe(self, played_action, context, reward):
        """
        Doesn't support batches of rewards
        """
        with torch.no_grad():
            self.update = context * ((reward - 1)/self._probabilities[0][played_action] - 1)
            self._weights[:,played_action:played_action+1] += self.update.squeeze(0).unsqueeze(1)
            
            
class LinUCB():
    """
    Implementation of the LinUCB algorithm with dijoint linear model (https://arxiv.org/pdf/1003.0146.pdf)
    """
    
    def __init__(self, K, D, inversion_interval = 100):
        """
        The inversion of the matrix is an expensive task an will be performed only every "inversion_interval" iterations
        """

        self._K = K
        self._D = D
        self._inversion_interval = inversion_interval
        self.reset()

    def select(self, context):
        """
        Selection the most optimal action using an upper confidence bound
        """
        with torch.no_grad():
            value = torch.Tensor([context.mm(i_theta) for i_theta in self._theta])
            confidence_interval = torch.stack([torch.sqrt(context.mm(self._A_inv[k]).mm(context.transpose(0,1))) for k in range(self._K)]).reshape(len(context), self._K)
            _, best_action = torch.max(value + confidence_interval, 1)
            return int(best_action[0])

    def observe(self, played_arm, context, reward, update = False):
        """
        Update of the counts. Predictions will be updated only after calling self._invert()
        """
        with torch.no_grad():
            self._A[played_arm] = self._A[played_arm] + context.transpose(0,1).mm(context)
            self._b[played_arm] = self._b[played_arm] + torch.transpose(context * reward, 0, 1)
            if update or ((self.t+1)%self._inversion_interval == 0):
                self._invert()

            self.t += 1

    def reset(self):
        
        self.t = 0
        self._A = [torch.eye(self._D) for k in range(self._K)]
        self._b = [torch.zeros((self._D, 1)) for k in range(self._K)]

        self._invert()

    def _invert(self):
        """
        Matrix inversion to update the model
        """
        with torch.no_grad():
            self._A_inv = [a.inverse() for a in self._A]
            self._theta = [self._A_inv[k].mm(self._b[k]) for k in range(self._K)]
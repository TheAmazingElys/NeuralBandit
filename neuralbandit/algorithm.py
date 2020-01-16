import torch

def get_simple_network(context_dimension, layer_count, layer_size):
    """
    Simple Network used in the NeuralBandit paper (https://hal.archives-ouvertes.fr/hal-01117311/document)
    """
    last_dim = context_dimension
    layers = []

    for i in range(layer_count):
        layers.append(nn.Linear(last_dim, layer_size))
        layers.append(nn.Sigmoid())
        last_dim = layer_size

    layers.append(nn.Linear(last_dim, 1))
    layers.append(nn.Sigmoid())

    return nn.Sequential(*layers)
    

class NeuralBandit():
    """
    Implementation of the NeuralBandit paper (https://hal.archives-ouvertes.fr/hal-01117311/document)
    """
    def __init__(self, action_count, context_dimension, layer_count = 0, layer_size = 0, lr = 0.1, gamma = 0.05, get_network = get_simple_network):
        
        self._K = action_count
        self._D = context_dimension
        self._gamma = gamma
        
        self._nn = [get_network(context_dimension, layer_count, layer_size) for k in range(self._K)]
        self._opt = torch.optim.SGD(list(itertools.chain(*[list(i_nn.parameters()) for i_nn in self._nn])), lr=lr, momentum=0.9)
        self._loss = torch.nn.MSELoss()
                
    def select(self, context):
        
        self._scores =  torch.stack([i_nn(context) for i_nn in self._nn]).reshape(len(context), self._K)
        _, self.best_actions = torch.max(self._scores, 1)

        mask = torch.zeros(self._scores.size()).scatter (1, self.best_actions.unsqueeze(1), 1.0)
        self._probabilities = (1 - self._gamma) * mask + torch.zeros((len(self._scores), self._K)) + self._gamma/self._K

        k = [np.random.choice(self._K, p=p.data.numpy()/sum(p.data.numpy())) for p in self._probabilities]

        return k[0]
    
    def observe(self, played_action, context, reward):
        """
        Doesn't support batches of rewards
        """
        loss = self._loss(self._scores[0][played_action], torch.tensor(reward).float())
        self._opt.zero_grad()
        loss.backward()
        self._opt.step()

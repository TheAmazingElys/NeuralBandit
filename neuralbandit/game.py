from torch.utils.data import DataLoader
from .data  import ContextualBanditDataset

class ContextualBanditGame():

    def __init__(self, dataset, T):
        
        assert isinstance(dataset, ContextualBanditDataset)
        
        self._dataset = dataset
        self.T = T
        self.init_game()
        
    @property
    def optimal_accuracy(self): 
        return self._dataset.optimal_accuracy

    def init_game(self):
        self._generator = self._get_generator()
        _ = self.get_context()    
        
    def _get_generator(self):
        """
        Return a generator yielding the contexts and updating the label
        """
        
        self._dataloader = DataLoader(self._dataset, batch_size=1,
                        shuffle=True)
        self._t = 0
        self._context = None
        self._label = None
        self._action_played = False
        
        while self._t < self.T: # We go through the dataloader only if the game is not over
            for i_context, i_label in self._dataloader:
                if self._t >= self.T: # If the game is over, the game just return None everywhere and we break the loop
                    self._context = None
                    self._label = None
                    break
                else:
                    self._action_played = False # We wait for the new action to be played
                    self._context = i_context # We update the current context and label
                    self._label = i_label
                    
                # We yield the current context, the label is not provided and will be used to compute the reward in the play function
                yield self._context
                
                while not self._action_played:# Return the same context while the next action is unplayed
                    yield self._context
                    
                self._t = self._t+1
                    
    def get_context(self):
        """
        Return the next context, None if the game is over
        """
        try:
            return next(self._generator)
        except:
            return None
    
    def play(self, action):
        """ Play an action and receive a reward. The reward is None if the game is already over """
        if self._label is None:
            return None
        
        self._action_played = True # We told the generator that an action has be played
        
        if self._label.data == action: #If the action match the label the reward is 1 else 0
            reward =  1
        else:
            reward =  0
            
        _ = self.get_context() # We go to the next steps, just in case the player is playing actions without requesting a new context
        
        return reward
from neuralbandit import __version__
from neuralbandit import get_cov_dataset, ContextualBanditGame, NeuralBandit
from neuralbandit.sota import RandomBandit, Banditron, LinUCB

def test_version():
    assert __version__ == '0.1.1'

def test_everything():

    dataset = get_cov_dataset()

    cumulative_reward = 0
    T = int(500)

    game = ContextualBanditGame(dataset, T)
    players = []

    players.append(RandomBandit(dataset.K))
    players.append(Banditron(dataset.K, dataset.D))
    players.append(LinUCB(dataset.K, dataset.D))
    players.append(NeuralBandit(dataset.K, dataset.D, layer_count = 1, layer_size = 64, gamma = 0.05))

    for i_player in players:
        for t in range(10):
            context = game.get_context()
            action = i_player.select(context)
            reward = game.play(action)
            cumulative_reward += reward
            i_player.observe(action, context, reward)
            
    cumulative_regret = (T*game.optimal_accuracy - cumulative_reward)

    assert True # \o/ We didn't crashed!
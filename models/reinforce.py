from torch.nn import Module, Linear, Flatten, ReLU, Sequential


class ReinforcePolicy(Module):
    def __init__(self, state_dim, n_actions, n_hidden):
        super(ReinforcePolicy, self).__init__()
        self.state_dim = state_dim
        self.n_actions = n_actions

        self.model = Sequential(
            Flatten(),
            Linear(state_dim, n_hidden),
            ReLU(),
            Linear(n_hidden, n_actions)
        )

    def forward(self, x):
        return self.model(x)

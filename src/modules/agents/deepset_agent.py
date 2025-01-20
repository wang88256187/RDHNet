from functools import partial
from math import sqrt
import torch as th
import torch.nn as nn
import torch.nn.functional as F


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    if module.bias is not None:
        bias_init(module.bias.data)
    return module

def init_(m, gain=0.01, activate=False):
    if activate:
        gain = nn.init.calculate_gain('relu')
    return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=gain)


class HyperLinear(nn.Module):
    """
    Linear network layers that allows for two additional complications:
        - parameters admit to be connected via a hyper-network like structure
        - network weights are transformed according to some rule before application
    """

    def __init__(self, in_size, out_size, use_hypernetwork=True):
        super(HyperLinear, self).__init__()

        self.use_hypernetwork = use_hypernetwork

        if not self.use_hypernetwork:
            self.w = nn.Linear(in_size, out_size)
        self.b = nn.Parameter(th.randn(out_size))

        # initialize layers
        stdv = 1. / sqrt(in_size)
        if not self.use_hypernetwork:
            self.w.weight.data.uniform_(-stdv, stdv)
        self.b.data.uniform_(-stdv, stdv)

        pass

    def forward(self, inputs, weights=None, weight_mod="abs", hypernet=None, **kwargs):
        """
        we assume inputs are of shape [a*bs*t]*v
        """
        assert inputs.dim() == 2, "we require inputs to be of shape [a*bs*t]*v"

        if self.use_hypernetwork:
            assert weights is not None, "if using hyper-network, need to supply the weights!"
            w = weights
        else:
            w = self.w.weight

        weight_mod_fn = None
        if weight_mod in ["abs"]:
            weight_mod_fn = th.abs
        elif weight_mod in ["pow"]:
            exponent = kwargs.get("exponent", 2)
            weight_mod_fn = partial(th.pow, exponent=exponent)
        elif callable(weight_mod):
            weight_mod_fn = weight_mod

        if weight_mod_fn is not None:
            w = weight_mod_fn(w)

        x = th.mm(inputs, w.t()) + self.b # TODO: why not BMM?
        return x

class DimenetAgent(nn.Module):
    def __init__(self, input_shape):
        super(DimenetAgent, self).__init__()
        self.args = args
        num_inputs = input_shape + args.n_actions
        hidden_size = args.rnn_hidden_dim

        self.fc1 = nn.Linear(num_inputs, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def get_weight_decay_weights(self):
        return {}

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state, actions):
        if actions is not None:
            inputs = th.cat([inputs, actions.contiguous().view(-1, actions.shape[-1])], dim=-1)
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return {"Q":q, "hidden_state": x}

class DeepsetAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(DeepsetAgent, self).__init__()
        self.args = args

        self.own_feat_dim = 4
        self.ally_feat_dim = 2
        self.enemy_feat_dim = 4
        self.landmark_dim = 2
        self.hidden_dim = 256

        self.n_agents = 3
        self.n_allies = self.n_agents - 1
        self.n_enemies = 1
        self.n_landmark = 2
        self.n_actions = 2

        self.obs_agent_id = False
        self.obs_last_action = False

        if self.obs_agent_id:
            # embedding table for agent_id
            self.agent_id_embedding = th.nn.Embedding(self.n_agents, self.hidden_dim)

        if self.obs_last_action:
            # embedding table for action id
            self.last_action_embedding = nn.Sequential(nn.LayerNorm(self.n_actions),
                                     init_(nn.Linear(self.n_actions, self.hidden_dim), activate=True))


        self.fc_own = nn.Sequential(
                                     init_(nn.Linear(self.own_feat_dim, self.hidden_dim), activate=True),

                                     init_(nn.Linear(self.hidden_dim, self.hidden_dim)))

        self.fc_ally = nn.Sequential(
                                     init_(nn.Linear(self.ally_feat_dim, self.hidden_dim), activate=True),

                                     init_(nn.Linear(self.hidden_dim, self.hidden_dim)))

        self.fc_enemy = nn.Sequential(
                                     init_(nn.Linear(self.enemy_feat_dim, self.hidden_dim), activate=True),

                                     init_(nn.Linear(self.hidden_dim, self.hidden_dim)))

        self.fc_landmark = nn.Sequential(
                                     init_(nn.Linear(self.landmark_dim, self.hidden_dim), activate=True),

                                     init_(nn.Linear(self.hidden_dim, self.hidden_dim)))

        self.fc_action = nn.Sequential(
                                     init_(nn.Linear(self.n_actions, self.hidden_dim), activate=True))

        self.fc_output = nn.Sequential(
                                     init_(nn.Linear(self.hidden_dim + 2, self.hidden_dim * 2), activate=True),

                                     init_(nn.Linear(self.hidden_dim * 2, 1)))



    def get_weight_decay_weights(self):
        return {}

    def init_hidden(self):
        # make hidden states on same device as model
        return th.zeros(1, 64).cuda()

    def data_processing(self, inputs):
        """
            inputs: ob , last action, agent id onehot
                ob:
                    vel: 2
                    pos: 2
                    landmark: n_landmark * 2
                    other agent pos: (num_good_agents + num_adversaries - 1) * 2
                    good agent vel: num_good_agents * 2


        """
        split_list = [16,2,3]
        bs = inputs.shape[0]

        # ob, last_ac, agent_id = inputs.split(split_list, dim=-1)
        # agent_id = agent_id.argmax(-1)
        ob = inputs
        last_ac = None
        agent_id = None
        own_feats = ob[:,:4]
        landmark_feats = ob[:,4:8].reshape(-1,self.n_landmark, self.landmark_dim)
        ally_feats = ob[:,8:12].reshape(-1, 2, self.ally_feat_dim)
        enemy_feats = ob[:,12:].reshape(-1, 1, self.enemy_feat_dim)
        # enemy_feats = ob[:, 12:14].reshape(-1, 1, 2)

        return bs, own_feats, landmark_feats, ally_feats, enemy_feats, last_ac, agent_id


    def forward(self, inputs, hidden_state, actions):
        """
            inputs:
                    ob: 16
                    action: 2
                    id: n_agent
        """
        bs, own_feats, landmark_feats, ally_feats, enemy_feats, last_action_indices, agent_id \
            = self.data_processing(inputs)

        # (1) Own embeddings
        embedding_own = self.fc_own(own_feats)
        # (2) ID embeddings and last actions
        if self.obs_agent_id:
            embedding_own = embedding_own + self.agent_id_embedding(agent_id).view(-1, self.hidden_dim)
        if self.obs_last_action:
            if last_action_indices is not None:  # t != 0
                embedding_own = embedding_own + self.last_action_embedding(last_action_indices).view(
                    -1, self.hidden_dim)

        embedding_own = embedding_own.unsqueeze(-2)

        # (3) landmark embeddings
        embedding_landmark = self.fc_landmark(landmark_feats)

        # (4) ally embeddings
        embedding_ally = self.fc_ally(ally_feats)

        # (5) enemy embeddings
        embedding_enemy = self.fc_enemy(enemy_feats)

        # (6) action embeddings
        # embedding_actions = self.fc_action(actions.contiguous().view(-1, actions.shape[-1]))

        # aggregate all ob embedding
        aggregated_ob_embedding = th.cat([embedding_own, embedding_landmark, embedding_ally, embedding_enemy], dim=-2).sum(-2)
        final_in = F.relu(aggregated_ob_embedding)
        # concat ob and action embedding
        x = th.cat([final_in, actions.contiguous().view(-1, actions.shape[-1])], dim=-1)
        q = self.fc_output(x)
        return {"Q": q, "hidden_state": x}



if __name__ == "__main__":

    inputs, actions = th.load("data.pth")
    agent = DeepsetAgent(21).cuda()
    output = agent(inputs, None, actions)

    print("ok")
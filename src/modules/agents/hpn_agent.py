import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from envs.particle.scenarios.maps_info import maps_info

class Merger(nn.Module):
    def __init__(self, head, hidden_dim):
        super(Merger, self).__init__()
        self.head = head
        if head > 1:
            self.weight = Parameter(torch.Tensor(1, head, hidden_dim).fill_(1.))
            self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        if self.head > 1:
            return torch.sum(self.softmax(self.weight) * x, dim=1, keepdim=False)
        else:
            return torch.squeeze(x, dim=1)


class HPNActor(nn.Module):
    """docstring for HPNActor"""

    def __init__(self, input_shape, args):
        super(HPNActor, self).__init__()
        self.args = args
        # self.obs_shape = args.obs_shape[0]
        # self.action_shape = args.action_shape[0]
        self.map_info = maps_info[args.env_args["scenario_name"]]
        self.self_dim = 4
        self.ally_dim = 4
        self.landmark_dim = 2
        self.adv_dim = 4
        # self.num_ally = 2
        # self.num_landmark = 2
        # self.num_adv = 1
        self.num_ally = self.map_info["n_agents"] - 1
        self.num_adv = self.map_info["n_enemies"]
        self.num_landmark = self.map_info["n_landmarks"]


        hyper_dim = 64
        hidden_dim = 64
        self.head = 4
        self.hidden_dim = hidden_dim
        # self, unique features, do not need hyper network
        self.self_fc = nn.Linear(self.self_dim, hidden_dim)
        # ally
        self.hyper_ally = nn.Sequential(
            nn.Linear(self.ally_dim, hyper_dim),
            nn.LeakyReLU(),
            nn.Linear(hyper_dim, (self.ally_dim + 1) * hidden_dim * self.head)
        )

        # adv
        if self.num_adv > 0:
            self.hyper_adv = nn.Sequential(
                nn.Linear(self.adv_dim, hyper_dim),
                nn.LeakyReLU(),
                nn.Linear(hyper_dim, (self.adv_dim + 1) * hidden_dim * self.head)
            )

        # landmark
        self.hyper_landmark = nn.Sequential(
            nn.Linear(self.landmark_dim, hyper_dim),
            nn.LeakyReLU(),
            nn.Linear(hyper_dim, (self.landmark_dim + 1) * hidden_dim * self.head)
        )


        # self.self_fc = nn.Linear(self.self_dim, hidden_dim)
        # self.hyper_ally = nn.Linear(self.ally_dim, (self.ally_dim + 1) * hidden_dim * self.head)
        # self.hyper_adv = nn.Linear(self.adv_dim, (self.adv_dim + 1) * hidden_dim * self.head)
        # output
        self.merger = Merger(self.head, hidden_dim)
        self.out_fc1 = nn.Linear(hidden_dim+2, hidden_dim)
        self.out_fc2 = nn.Linear(hidden_dim, 1)

    def init_hidden(self):
        # make hidden states on same device as model
        return torch.zeros(1, 64).cuda()

    def forward(self, inputs, hidden_state, actions):
        self_feats, ally_feats, adv_feats, landmark_feats, last_ac, agent_id = self._build_input(inputs)
        # self
        self_out = self.self_fc(self_feats)  # (bz, hidden_dim)
        # ally
        hyper_ally_out = self.hyper_ally(ally_feats)  # (bz, num_ally, (ally_dim+1)*hidden_dim*head)
        # (bz*num_ally, ally_dim, hidden_dim)
        hyper_ally_w = hyper_ally_out[:, :, :self.ally_dim * self.hidden_dim*self.head].view(-1, self.ally_dim, self.hidden_dim*self.head)
        hyper_ally_b = hyper_ally_out[:, :, self.ally_dim * self.hidden_dim*self.head:].view(-1, 1, self.hidden_dim*self.head)
        ally_feats = ally_feats.reshape(-1, 1, self.ally_dim)  # (bz*num_ally, 1, ally_dim)
        ally_out = torch.matmul(ally_feats, hyper_ally_w) + hyper_ally_b
        ally_out = ally_out.view(-1, self.num_ally, self.head, self.hidden_dim)
        # ally_mask = ally_mask.expand(-1, -1, self.head, self.hidden_dim)
        # ally_out = ally_out.masked_fill(ally_mask, 0.)
        ally_out = ally_out.sum(dim=1, keepdim=False)  # (bz, head, hidden_dim)

        # adv
        if self.num_adv > 0:
            hyper_adv_out = self.hyper_adv(adv_feats)  # (bz, num_adv, (adv_dim+1)*hidden_dim*head)
            # (bz*num_adv, adv_dim, hidden_dim)
            hyper_adv_w = hyper_adv_out[:, :, :self.adv_dim * self.hidden_dim*self.head].view(-1, self.adv_dim, self.hidden_dim*self.head)
            hyper_adv_b = hyper_adv_out[:, :, self.adv_dim * self.hidden_dim*self.head:].view(-1, 1, self.hidden_dim*self.head)
            adv_feats = adv_feats.reshape(-1, 1, self.adv_dim)
            adv_out = torch.matmul(adv_feats, hyper_adv_w) + hyper_adv_b
            adv_out = adv_out.view(-1, self.num_adv, self.head, self.hidden_dim)
            # adv_mask = adv_mask.expand(-1, -1, self.head, self.hidden_dim)
            # adv_out = adv_out.masked_fill(adv_mask, 0.)
            adv_out = adv_out.sum(dim=1, keepdim=False)     # (bz, head, hidden_dim)

        # landmark
        hyper_landmark_out = self.hyper_landmark(landmark_feats)  # (bz, num_ally, (ally_dim+1)*hidden_dim*head)
        # (bz*num_ally, ally_dim, hidden_dim)
        hyper_landmark_w = hyper_landmark_out[:, :, :self.landmark_dim * self.hidden_dim * self.head].view(-1, self.landmark_dim,
                                                                                               self.hidden_dim * self.head)
        hyper_landmark_b = hyper_landmark_out[:, :, self.landmark_dim * self.hidden_dim * self.head:].view(-1, 1,
                                                                                               self.hidden_dim * self.head)
        landmark_feats = landmark_feats.reshape(-1, 1, self.landmark_dim)
        landmark_out = torch.matmul(landmark_feats, hyper_landmark_w) + hyper_landmark_b
        landmark_out = landmark_out.view(-1, self.num_landmark, self.head, self.hidden_dim)
        landmark_out = landmark_out.sum(dim=1, keepdim=False)


        # merge all info
        # final_in = torch.cat([self_out, ally_out, adv_out], dim=-1)
        tmp = ally_out + landmark_out
        if self.num_adv > 0:
            tmp = tmp + adv_out
        final_in = self_out + self.merger(tmp)
        out = F.relu(final_in)
        out = torch.cat([out, actions.contiguous().view(-1, actions.shape[-1])], dim=-1)

        x = F.relu(self.out_fc1(out))
        q = self.out_fc2(x)
        return {"Q": q, "hidden_state": x}

    def _build_input(self, inputs):
        # split_list = [16, 2, 3]

        # bs = inputs.shape[0]
        #
        # ob, last_ac, agent_id = inputs.split(split_list, dim=-1)
        # agent_id = agent_id.argmax(-1)
        # # ob = inputs
        # # last_ac = None
        # # agent_id = None
        # self_feats = ob[:, :4]
        # landmark_feats = ob[:, 4:8].reshape(-1, self.num_landmark, self.landmark_dim)
        # ally_feats = ob[:, 8:12].reshape(-1, 2, self.ally_dim)
        # adv_feats = ob[:, 12:].reshape(-1, self.num_adv, self.adv_dim)

        bs = inputs.shape[0]

        # ob, last_ac, agent_id = inputs.split(split_list, dim=-1)
        # agent_id = agent_id.argmax(-1)
        adv_feats = None
        ob = inputs
        last_ac = None
        agent_id = None
        ind = 4
        self_feats = ob[:, :ind]
        ind1 = ind + self.num_landmark * self.landmark_dim
        landmark_feats = ob[:, ind:ind1].reshape(-1, self.num_landmark, self.landmark_dim)
        ind2 = ind1 + self.num_ally * 2
        ally_pos = ob[:, ind1:ind2].reshape(-1, self.num_ally, 2)
        ind3 = ind2 + self.num_adv * 2
        if self.num_adv > 0:
            adv_pos = ob[:, ind2:ind3].reshape(-1, self.num_adv, 2)
        ind4 = ind3 + self.num_ally * 2
        ally_vel = ob[:, ind3:ind4].reshape(-1, self.num_ally, 2)
        ind5 = ind4 + self.num_adv * 2
        if self.num_adv > 0:
            adv_vel = ob[:, ind4:ind5].reshape(-1, self.num_adv, 2)
            adv_feats = torch.cat([adv_pos, adv_vel], dim=-1)
        ally_feats = torch.cat([ally_pos, ally_vel], dim=-1)
        return self_feats, ally_feats, adv_feats, landmark_feats, last_ac, agent_id


class HPNCritic(nn.Module):
    """docstring for HPNCritic"""

    def __init__(self, args):
        super(HPNCritic, self).__init__()
        self.args = args
        self.obs_shape = args.obs_shape[0]
        self.action_shape = args.action_shape[0]

        self.self_dim = 7
        self.ally_dim = 5
        self.adv_dim = 6
        self.num_ally = args.max_num_RUAVs - 1
        self.num_adv = args.max_num_BUAVs

        hyper_dim = 128
        hidden_dim = 256
        self.head = 4
        self.hidden_dim = hidden_dim
        # self, unique features, do not need hyper network
        self.self_fc = nn.Linear(self.self_dim + self.action_shape, hidden_dim)
        # ally
        self.hyper_ally = nn.Sequential(
            nn.Linear(self.ally_dim + self.action_shape, hyper_dim),
            nn.LeakyReLU(),
            nn.Linear(hyper_dim, (self.ally_dim + self.action_shape + 1) * hidden_dim*self.head)
        )
        # adv
        self.hyper_adv = nn.Sequential(
            nn.Linear(self.adv_dim, hyper_dim),
            nn.LeakyReLU(),
            nn.Linear(hyper_dim, (self.adv_dim + 1) * hidden_dim * self.head)
        )
        # self.self_fc = nn.Linear(self.self_dim+self.action_shape, hidden_dim)
        # self.hyper_ally = nn.Linear(self.ally_dim+self.action_shape, (self.ally_dim+self.action_shape + 1)*hidden_dim*self.head)
        # self.hyper_adv = nn.Linear(self.adv_dim, (self.adv_dim + 1)*hidden_dim*self.head)
        # output
        self.merger = Merger(self.head, hidden_dim)
        self.out_fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.out_fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action, agent_id):
        self_feats, ally_feats, adv_feats, ally_mask, adv_mask = self._build_input(state, action, agent_id)
        # self
        self_out = self.self_fc(self_feats)  # (bz, hidden_dim)
        # ally
        hyper_ally_out = self.hyper_ally(ally_feats)  # (bz, num_ally, (ally_dim+1)*hidden_dim*head)
        # (bz*num_ally, ally_dim, hidden_dim)
        hyper_ally_w = hyper_ally_out[:, :, :(self.ally_dim + self.action_shape) * self.hidden_dim*self.head].view(-1,
                                                                                                         self.ally_dim + self.action_shape,
                                                                                                         self.hidden_dim*self.head)
        hyper_ally_b = hyper_ally_out[:, :, (self.ally_dim + self.action_shape) * self.hidden_dim*self.head:].view(-1, 1,
                                                                                                         self.hidden_dim*self.head)
        ally_feats = ally_feats.reshape(-1, 1, self.ally_dim + self.action_shape)
        ally_out = torch.matmul(ally_feats, hyper_ally_w) + hyper_ally_b
        ally_out = ally_out.view(-1, self.num_ally, self.head, self.hidden_dim)
        ally_mask = ally_mask.expand(-1, -1, self.head, self.hidden_dim)
        ally_out = ally_out.masked_fill(ally_mask, 0.)
        ally_out = ally_out.sum(dim=1, keepdim=False)  # (bz, head, hidden_dim)
        # adv
        hyper_adv_out = self.hyper_adv(adv_feats)  # (bz, num_adv, (adv_dim+1)*hidden_dim*head)
        # (bz*num_adv, adv_dim, hidden_dim)
        hyper_adv_w = hyper_adv_out[:, :, :self.adv_dim * self.hidden_dim*self.head].view(-1, self.adv_dim, self.hidden_dim*self.head)
        hyper_adv_b = hyper_adv_out[:, :, self.adv_dim * self.hidden_dim*self.head:].view(-1, 1, self.hidden_dim*self.head)
        adv_feats = adv_feats.reshape(-1, 1, self.adv_dim)
        adv_out = torch.matmul(adv_feats, hyper_adv_w) + hyper_adv_b
        adv_out = adv_out.view(-1, self.num_adv, self.head, self.hidden_dim)
        adv_mask = adv_mask.expand(-1, -1, self.head, self.hidden_dim)
        # adv_out = adv_out.masked_fill(adv_mask, 0.)
        adv_out = adv_out.sum(dim=1, keepdim=False)     # (bz, head, hidden_dim)
        # merge all info
        # final_in = torch.cat([self_out, ally_out, adv_out], dim=-1)
        final_in = self_out + self.merger(ally_out + adv_out)
        out = F.relu(final_in)
        out = F.relu(self.out_fc1(out))
        out = self.out_fc2(out)
        return out

    def _build_input(self, state, action, agent_id):
        action = torch.stack(action, dim=1)  # (bz, num_agents, action_shape)
        self_feats, ally_feats, land_feats, \
            ally_mask, adv_mask = state.split([self.self_dim,
                                               self.num_ally * self.ally_dim,
                                               self.num_adv * self.adv_dim,
                                               self.num_ally,
                                               self.num_adv],
                                              dim=-1)
        self_feats = torch.cat([self_feats, action[:, agent_id]], dim=-1)  # (bz, dim)
        ally_feats = ally_feats.reshape(-1, self.num_ally, self.ally_dim)
        other_actions = action[:, torch.arange(action.shape[1]) != agent_id]  # (bz, num_ally, action_shape)
        ally_feats = torch.cat([ally_feats, other_actions], dim=-1)  # (bz, num_ally, dim)
        adv_feats = land_feats.reshape(-1, self.num_adv, self.adv_dim)
        ally_death_mask = ally_mask.reshape(-1, self.num_ally, 1, 1) < 0.5
        adv_death_mask = adv_mask.reshape(-1, self.num_adv, 1, 1) < 0.5
        return self_feats, ally_feats, adv_feats, ally_death_mask, adv_death_mask

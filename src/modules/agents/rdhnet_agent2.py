import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from modules.layers.basis_layers import rbf_class_mapping
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
            return torch.sum(self.softmax(self.weight) * x, dim=-2, keepdim=False)
        else:
            return torch.squeeze(x, dim=1)


class RDHAgent(nn.Module):
    """docstring for HPNActor"""

    def __init__(self, input_shape, args):
        super(RDHAgent, self).__init__()
        self.args = args
        # self.obs_shape = args.obs_shape[0]
        # self.action_shape = args.action_shape[0]
        self.map_info = maps_info[args.env_args["scenario_name"]]

        self.n_agents = self.map_info["n_agents"]
        self.own_feat_dim = 4
        self.ally_feat_dim = 2
        self.landmark_dim = 2
        self.enemy_feat_dim = 4

        self.n_allies = self.n_agents - 1
        self.n_enemies = self.map_info["n_enemies"]
        self.n_landmark = self.map_info["n_landmarks"]
        self.n_entities = self.n_agents + self.n_enemies + self.n_landmark
        self.hyper_input_dim = 9

        self.n_rbf = 3

        hyper_dim = 64
        hidden_dim = 64
        self.head = 4
        self.hidden_dim = hidden_dim
        # self, unique features, do not need hyper network
        self.self_fc = nn.Linear(self.own_feat_dim, hidden_dim)

        # ally
        self.hyper_ally = nn.Sequential(
            nn.Linear(self.hyper_input_dim, hyper_dim),
            nn.LeakyReLU(),
            nn.Linear(hyper_dim, (self.hyper_input_dim + 1) * hidden_dim * self.head)
        )

        # enemy
        if self.n_agents > 0:
            self.hyper_enemy = nn.Sequential(
                nn.Linear(self.hyper_input_dim, hyper_dim),
                nn.LeakyReLU(),
                nn.Linear(hyper_dim, (self.hyper_input_dim + 1) * hidden_dim * self.head)
            )

        # landmark
        self.hyper_landmark = nn.Sequential(
            nn.Linear(self.hyper_input_dim, hyper_dim),
            nn.LeakyReLU(),
            nn.Linear(hyper_dim, (self.hyper_input_dim + 1) * hidden_dim * self.head)
        )

        self.rbf_fn = rbf_class_mapping["gauss"](
            num_rbf=self.n_rbf,
            rbound_upper=10
        )

        # self.self_fc = nn.Linear(self.own_feat_dim, hidden_dim)
        # self.hyper_ally = nn.Linear(self.ally_feat_dim, (self.ally_feat_dim + 1) * hidden_dim * self.head)
        # self.hyper_adv = nn.Linear(self.enemy_feat_dim, (self.enemy_feat_dim + 1) * hidden_dim * self.head)
        # output
        self.merger = Merger(self.head, hidden_dim)
        self.out_fc1 = nn.Linear(hidden_dim + 3, hidden_dim)
        self.out_fc2 = nn.Linear(hidden_dim, 1)

    def init_hidden(self):
        # make hidden states on same device as model
        return torch.zeros(1, 64).cuda()

    def forward(self, inputs, hidden_state, actions, cem_sample=False):
        bs, own_feats, entity_pos, entity_relative_vel, \
            entity_dist, entity_vel_norm, angle_info, actions_norm, actions_angle_info = self._build_input(inputs,
                                                                                                           actions,cem_sample)

        # m_ijk
        feat_dist = self.rbf_fn(entity_dist).unsqueeze(-3).repeat(1, 1, self.n_entities - 1, 1, 1)
        feat_angle = angle_info
        feat_node_feats = entity_relative_vel

        feat_inputs = torch.cat([feat_dist, feat_angle, feat_node_feats], dim=-1)

        # ally
        hyper_ally_out = self.hyper_ally(feat_inputs)
        hyper_ally_w = hyper_ally_out[:, :, :, :, :self.hyper_input_dim * self.hidden_dim * self.head].view(-1,
                                                                                                            self.hyper_input_dim,
                                                                                                            self.hidden_dim * self.head)
        hyper_ally_b = hyper_ally_out[:, :, :, :, self.hyper_input_dim * self.hidden_dim * self.head:].view(-1, 1,
                                                                                                            self.hidden_dim * self.head)

        ally_feats = feat_inputs.reshape(-1, 1, self.hyper_input_dim)
        ally_out = torch.matmul(ally_feats, hyper_ally_w) + hyper_ally_b
        ally_out = ally_out.view(-1, self.n_agents, self.n_entities - 1, self.n_entities - 1, self.head,
                                 self.hidden_dim)

        # enemy
        if self.n_enemies > 0:
            hyper_enemy_out = self.hyper_enemy(feat_inputs)
            hyper_enemy_w = hyper_enemy_out[:, :, :, :, :self.hyper_input_dim * self.hidden_dim * self.head].view(-1,
                                                                                                                  self.hyper_input_dim,
                                                                                                                  self.hidden_dim * self.head)
            hyper_enemy_b = hyper_enemy_out[:, :, :, :, self.hyper_input_dim * self.hidden_dim * self.head:].view(-1, 1,
                                                                                                                  self.hidden_dim * self.head)

            enemy_feats = feat_inputs.reshape(-1, 1, self.hyper_input_dim)
            enemy_out = torch.matmul(enemy_feats, hyper_enemy_w) + hyper_enemy_b
            enemy_out = enemy_out.view(-1, self.n_agents, self.n_entities - 1, self.n_entities - 1, self.head,
                                       self.hidden_dim)

        # landmark
        hyper_landmark_out = self.hyper_landmark(feat_inputs)
        hyper_landmark_w = hyper_landmark_out[:, :, :, :, :self.hyper_input_dim * self.hidden_dim * self.head].view(-1,
                                                                                                                    self.hyper_input_dim,
                                                                                                                    self.hidden_dim * self.head)
        hyper_landmark_b = hyper_landmark_out[:, :, :, :, self.hyper_input_dim * self.hidden_dim * self.head:].view(-1,
                                                                                                                    1,
                                                                                                                    self.hidden_dim * self.head)

        landmark_feats = feat_inputs.reshape(-1, 1, self.hyper_input_dim)
        landmark_out = torch.matmul(landmark_feats, hyper_landmark_w) + hyper_landmark_b
        landmark_out = landmark_out.view(-1, self.n_agents, self.n_entities - 1, self.n_entities - 1, self.head,
                                         self.hidden_dim)

        m_jki = torch.zeros_like(ally_out)
        m_jki[:, :, :, :self.n_allies] = ally_out[:, :, :, :self.n_allies]
        if self.n_enemies > 0:
            m_jki[:, :, :, self.n_allies:self.n_allies + self.n_enemies] = enemy_out[:, :, :,
                                                                           self.n_allies:self.n_allies + self.n_enemies]
        m_jki[:, :, :, self.n_allies + self.n_enemies:] = landmark_out[:, :, :, self.n_allies + self.n_enemies:]
        # index = torch.arange(self.n_entities-1).long()
        # m_jki[:,:,index,index] = 0
        m_jki_ = self.merger(m_jki.sum(3))
        if cem_sample:
            m_jki_ = m_jki_.repeat(bs,1,1,1)
        actions_norm = actions_norm.repeat(1, 1, self.n_entities - 1).unsqueeze(-1)

        final_inputs = torch.cat([m_jki_, actions_norm, actions_angle_info], dim=-1)
        x = F.relu(final_inputs)
        x = F.relu(self.out_fc1(x))
        q = self.out_fc2(x).mean(-2).reshape(-1, 1)
        return {"Q": q, "hidden_state": x}

    def _build_input(self, inputs, actions, cem_sample=False):
        # split_list = [20, 2, 3]
        #
        # ob, last_ac, agent_id = inputs.split(split_list, dim=-1)
        # agent_id = agent_id.argmax(-1)


        if cem_sample:
            ob = inputs.reshape(-1, 64, self.n_agents, 18)[:,0].reshape(-1, 18)
            bs = int(inputs.shape[0] / (self.n_agents))
        else:
            ob = inputs
            bs = int(inputs.shape[0] / self.n_agents)

        ind = 4
        own_feats = ob[:, :ind].reshape(-1, self.n_agents, self.own_feat_dim)


        # entity relative pos (ally, landmark, enemy)
        ind1 = ind + self.n_landmark * self.landmark_dim
        landmark_pos = ob[:, ind:ind1].reshape(-1, self.n_agents, self.n_landmark, 2)
        ind2 = ind1 + self.n_allies * 2
        ally_pos = ob[:, ind1:ind2].reshape(-1, self.n_agents, self.n_allies, 2)
        ind3 = ind2 + self.n_enemies * 2
        if self.n_enemies > 0:
            enemy_pos = ob[:, ind2:ind3].reshape(-1, self.n_agents, self.n_enemies, 2)
            entity_pos = torch.cat([ally_pos, enemy_pos, landmark_pos], dim=-2)
        else:
            entity_pos = torch.cat([ally_pos, landmark_pos], dim=-2)

        # entity relative vel (ally, landmark, enemy)
        ind4 = ind3 + self.n_allies * 2
        ally_vel = ob[:, ind3:ind4].reshape(-1, self.n_agents, self.n_allies, 2)
        landmark_vel = torch.zeros_like(landmark_pos) - own_feats[:, :, :2].unsqueeze(
            -2)  # landmark' vel should be zero
        ind5 = ind4 + self.n_enemies * 2
        if self.n_enemies > 0:
            enemy_vel = ob[:, ind4:ind5].reshape(-1, self.n_agents, self.n_enemies, 2)
            entity_vel = torch.cat([ally_vel, enemy_vel, landmark_vel], dim=-2)
        else:
            entity_vel = torch.cat([ally_vel, landmark_vel], dim=-2)

        entity_relative_vel = entity_vel.unsqueeze(-3) - entity_vel.unsqueeze(-2)

        entity_vel_norm = (entity_vel ** 2).sum(-1).sqrt()
        sin_alpha = entity_vel[:, :, :, 1] / (entity_vel_norm + 1e-7)
        cos_alpha = entity_vel[:, :, :, 0] / (entity_vel_norm + 1e-7)

        sin_C = sin_alpha.unsqueeze(-2).repeat(1, 1, self.n_entities - 1, 1)
        cos_C = cos_alpha.unsqueeze(-2).repeat(1, 1, self.n_entities - 1, 1)

        # actions norm and angle
        actions = actions.view(bs, self.n_agents, 2)
        actions_norm = (actions ** 2).sum(-1).sqrt()
        sin_beta = actions[:, :, 1] / (actions_norm + 1e-7)
        cos_beta = actions[:, :, 0] / (actions_norm + 1e-7)

        sin_D = sin_beta.unsqueeze(-1).repeat(1, 1, self.n_entities - 1)
        cos_D = cos_beta.unsqueeze(-1).repeat(1, 1, self.n_entities - 1)

        # entities' distance from central agent
        ally_dist = (ally_pos ** 2).sum(-1).sqrt()
        landmark_dist = (landmark_pos ** 2).sum(-1).sqrt()
        if self.n_enemies > 0:
            enemy_dist = (enemy_pos ** 2).sum(-1).sqrt()
            entity_dist = torch.cat([ally_dist, enemy_dist, landmark_dist], dim=-1)
        else:
            entity_dist = torch.cat([ally_dist, landmark_dist], dim=-1)

        sin_theta = entity_pos[:, :, :, 1] / (entity_dist + 1e-7)
        cos_theta = entity_pos[:, :, :, 0] / (entity_dist + 1e-7)

        sin_A = sin_theta.unsqueeze(-2).repeat(1, 1, self.n_entities - 1, 1)
        sin_B = sin_theta.unsqueeze(-1).repeat(1, 1, 1, self.n_entities - 1)
        cos_A = cos_theta.unsqueeze(-2).repeat(1, 1, self.n_entities - 1, 1)
        cos_B = cos_theta.unsqueeze(-1).repeat(1, 1, 1, self.n_entities - 1)

        sin_A_minus_B = sin_A * cos_B - cos_A * sin_B
        cos_A_minus_B = cos_A * cos_B + sin_A * sin_B

        sin_C_minus_B = sin_C * cos_B - cos_C * sin_B
        cos_C_minus_B = cos_C * cos_B + sin_C * sin_B

        sin_D_minus_B = sin_D * cos_theta - cos_D * sin_theta
        cos_D_minus_B = cos_D * cos_theta + sin_D * sin_theta

        angle_info = torch.cat([sin_A_minus_B.unsqueeze(-1), cos_A_minus_B.unsqueeze(-1),
                                sin_C_minus_B.unsqueeze(-1), cos_C_minus_B.unsqueeze(-1)], dim=-1)

        actions_angle_info = torch.cat([sin_D_minus_B.unsqueeze(-1), cos_D_minus_B.unsqueeze(-1)], dim=-1)

        return bs, own_feats, entity_pos, entity_relative_vel, \
            entity_dist.unsqueeze(-1), entity_vel_norm.unsqueeze(-1), angle_info, actions_norm.unsqueeze(
            -1), actions_angle_info


if __name__ == "__main__":
    inputs, actions = torch.load("data.pth")
    agent = RDHAgent(None, None).cuda()
    output = agent(inputs, None, actions)

    print("ok")

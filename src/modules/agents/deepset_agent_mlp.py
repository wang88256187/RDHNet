import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from modules.layers.basis_layers import rbf_class_mapping

class DeepsetMLP(nn.Module):
    """docstring for HPNActor"""

    def __init__(self, input_shape, args):
        super(DeepsetMLP, self).__init__()
        self.args = args
        # self.obs_shape = args.obs_shape[0]
        # self.action_shape = args.action_shape[0]

        self.n_agents = 3
        self.own_feat_dim = 4
        self.ally_feat_dim = 2
        self.landmark_dim = 2
        self.enemy_feat_dim = 4

        self.n_agents = 3
        self.n_allies = self.n_agents - 1
        self.n_enemies = 1
        self.n_landmark = 2
        self.n_entities = self.n_agents + self.n_enemies + self.n_landmark
        self.input_dim = 9

        self.n_rbf = 3

        hidden_dim = 256

        self.hidden_dim = hidden_dim
        # self, unique features, do not need hyper network
        self.self_fc = nn.Linear(self.own_feat_dim, hidden_dim)

        # ally
        self.ally_fc = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        # enemy
        self.enemy_fc = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        # landmark
        self.landmark_fc = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.rbf_fn = rbf_class_mapping["gauss"](
            num_rbf=self.n_rbf,
            rbound_upper=10
        )

        # self.self_fc = nn.Linear(self.own_feat_dim, hidden_dim)
        # self.hyper_ally = nn.Linear(self.ally_feat_dim, (self.ally_feat_dim + 1) * hidden_dim * self.head)
        # self.hyper_adv = nn.Linear(self.enemy_feat_dim, (self.enemy_feat_dim + 1) * hidden_dim * self.head)
        # output
        # self.merger = Merger(self.head, hidden_dim)
        self.out_fc1 = nn.Linear(hidden_dim+3, hidden_dim)
        self.out_fc2 = nn.Linear(hidden_dim, 1)

    def init_hidden(self):
        # make hidden states on same device as model
        return torch.zeros(1, 64).cuda()

    def forward(self, inputs, hidden_state, actions):
        bs, own_feats, entity_pos, entity_relative_vel, \
            entity_dist, entity_vel_norm, angle_info, actions_norm, actions_angle_info = self._build_input(inputs, actions)

        # m_ijk
        feat_dist = self.rbf_fn(entity_dist).unsqueeze(-3).repeat(1,1,self.n_entities-1,1,1)
        feat_angle = angle_info
        feat_node_feats = entity_relative_vel

        feat_inputs = torch.cat([feat_dist,feat_angle,feat_node_feats],dim=-1)

        # ally
        ally_out = self.ally_fc(feat_inputs)

        # enemy
        enemy_out = self.enemy_fc(feat_inputs)

        # landmark
        landmark_out = self.landmark_fc(feat_inputs)

        # aggregate info
        m_jki = torch.zeros_like(ally_out)
        m_jki[:,:,:,:2] = ally_out[:,:,:,:2]
        m_jki[:,:,:,2:3] = enemy_out[:,:,:,2:3]
        m_jki[:,:,:,3:] = landmark_out[:,:,:,3:]

        m_jki_ = m_jki.sum(3)
        actions_norm = actions_norm.repeat(1,1,self.n_entities-1).unsqueeze(-1)
        final_inputs = torch.cat([m_jki_, actions_norm, actions_angle_info], dim=-1)
        x = F.relu(final_inputs)
        x = F.relu(self.out_fc1(x))
        q = self.out_fc2(x).mean(-2).reshape(-1,1)
        return {"Q": q, "hidden_state": x}

    def _build_input(self, inputs, actions):
        # split_list = [20, 2, 3]
        #
        # ob, last_ac, agent_id = inputs.split(split_list, dim=-1)
        # agent_id = agent_id.argmax(-1)
        ob = inputs

        own_feats = ob[:, :4].reshape(-1, self.n_agents, self.own_feat_dim)
        bs = own_feats.shape[0]

        # entity relative pos (ally, landmark, enemy)
        landmark_pos = ob[:, 4:8].reshape(-1, self.n_agents, self.n_landmark, 2)
        ally_pos = ob[:, 8:12].reshape(-1, self.n_agents, self.n_allies, 2)
        enemy_pos = ob[:, 12:14].reshape(-1, self.n_agents, self.n_enemies, 2)
        entity_pos = torch.cat([ally_pos, enemy_pos, landmark_pos], dim=-2)

        # entity relative vel (ally, landmark, enemy)
        ally_vel = ob[:, 14:18].reshape(-1, self.n_agents, self.n_allies, 2)
        enemy_vel = ob[:, 18:20].reshape(-1, self.n_agents, self.n_enemies, 2)
        landmark_vel = torch.zeros_like(landmark_pos) - own_feats[:,:,:2].unsqueeze(-2)  # landmark' vel should be zero
        entity_vel = torch.cat([ally_vel, enemy_vel, landmark_vel], dim=-2)

        entity_relative_vel = entity_vel.unsqueeze(-3) - entity_vel.unsqueeze(-2)

        entity_vel_norm = (entity_vel ** 2).sum(-1).sqrt()
        sin_alpha = entity_vel[:,:,:,1] / (entity_vel_norm + 1e-7)
        cos_alpha = entity_vel[:,:,:,0] / (entity_vel_norm + 1e-7)

        sin_C = sin_alpha.unsqueeze(-2).repeat(1,1,5,1)
        cos_C = cos_alpha.unsqueeze(-2).repeat(1,1,5,1)


        # actions norm and angle
        actions = actions.view(bs, self.n_agents, 2)
        actions_norm = (actions ** 2).sum(-1).sqrt()
        sin_beta = actions[:,:, 1] / (actions_norm + 1e-7)
        cos_beta = actions[:,:, 0] / (actions_norm + 1e-7)

        sin_D = sin_beta.unsqueeze(-1).repeat(1,1,5)
        cos_D = cos_beta.unsqueeze(-1).repeat(1, 1, 5)

        "/home/wangdongzi/Desktop/scaleable_MARL/facmac / src / main.py"
        # entities' distance from central agent
        ally_dist = (ally_pos ** 2).sum(-1).sqrt()
        landmark_dist = (landmark_pos ** 2).sum(-1).sqrt()
        enemy_dist = (enemy_pos ** 2).sum(-1).sqrt()
        entity_dist = torch.cat([ally_dist, enemy_dist, landmark_dist], dim=-1)

        sin_theta = entity_pos[:,:,:,1] / (entity_dist + 1e-7)
        cos_theta = entity_pos[:,:,:,0] / (entity_dist + 1e-7)

        sin_A = sin_theta.unsqueeze(-2).repeat(1,1,5,1)
        sin_B = sin_theta.unsqueeze(-1).repeat(1,1,1,5)
        cos_A = cos_theta.unsqueeze(-2).repeat(1,1,5,1)
        cos_B = cos_theta.unsqueeze(-1).repeat(1,1,1,5)

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
            entity_dist.unsqueeze(-1), entity_vel_norm.unsqueeze(-1),angle_info, actions_norm.unsqueeze(-1), actions_angle_info

if __name__ == "__main__":

    inputs, actions = torch.load("data.pth")
    agent = DeepsetMLP(None, None).cuda()
    output = agent(inputs, None, actions)

    print("ok")
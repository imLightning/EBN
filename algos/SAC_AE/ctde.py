import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from algos.SAC_AE.sac_ae import weight_init
from algos.SAC_AE.utils import soft_update_params


class CentralizedCritic(nn.Module):
    """MADDPG-style centralized critic.

    Input: concatenation of every agent's encoded feature, vector state and
    action (in agent order). Output: one Q value per agent, learned with
    parameter sharing across agents (team critic). Used only during training
    (CTDE); execution remains fully decentralized (each actor sees only its own
    observation).
    """
    def __init__(self, num_agents, encoder_feature_dim, robot_goal_state_dim,
                 action_dim, hidden_dim):
        super().__init__()
        input_dim = num_agents * (encoder_feature_dim + robot_goal_state_dim + action_dim)
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, num_agents)
        )
        self.apply(weight_init)

    def forward(self, x):
        return self.trunk(x)


def update_multi(agents, centralized_critic, centralized_critic_target,
                 critic_optimizer, replay_buffer, writer, step,
                 discount=0.99, critic_tau=0.005, encoder_tau=0.005,
                 actor_update_freq=2, critic_target_update_freq=2):
    """Joint CTDE update of all agents using a shared centralized critic.

    ``agents`` is a list of SacAeAgent; only their actors, critic encoders and
    temperatures are used here (the per-agent local critics are left unused in
    CTDE mode).
    """
    obs_list, state_list, action_list, rewards, next_obs_list, next_state_list, not_dones = replay_buffer.sample()
    num_agents = len(agents)

    # ---------------- centralized critic update ----------------
    with torch.no_grad():
        # target features from the *soft-updated* target encoders (stabilises
        # the Bellman target — the SAC-AE design always uses a target encoder)
        next_feats = [agents[i].critic_target.encoder(next_obs_list[i]) for i in range(num_agents)]
        next_actions = []
        next_log_pis = []
        for i in range(num_agents):
            _, na, logp, _ = agents[i].actor(next_obs_list[i], next_state_list[i])
            next_actions.append(na)
            next_log_pis.append(logp)
        target_in = torch.cat(next_feats + next_state_list + next_actions, dim=-1)
        target_q_all = centralized_critic_target(target_in)
        alpha_all = torch.stack([agents[i].alpha.detach() for i in range(num_agents)], dim=0)
        log_pi_all = torch.cat(next_log_pis, dim=-1)
        target_v = target_q_all - alpha_all * log_pi_all
        target_q = rewards + not_dones * discount * target_v

    feats = [agents[i].critic.encoder(obs_list[i]) for i in range(num_agents)]
    q_in = torch.cat(feats + state_list + action_list, dim=-1)
    q_all = centralized_critic(q_in)
    critic_loss = F.mse_loss(q_all, target_q)
    writer.add_scalar('train_critic/loss', critic_loss, step)
    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()

    # keep the actors' perception conv layers tied to the critics' encoders
    for i in range(num_agents):
        agents[i].actor.encoder.copy_conv_weights_from(agents[i].critic.encoder)

    # ---------------- per-agent actor + temperature update ----------------
    if step % actor_update_freq == 0:
        with torch.no_grad():
            feats_det = [f.detach() for f in feats]
        for i in range(num_agents):
            _, pi_i, log_pi_i, log_std_i = agents[i].actor(obs_list[i], state_list[i], detach_encoder=True)
            ordered_actions = []
            for j in range(num_agents):
                if j == i:
                    ordered_actions.append(pi_i)
                else:
                    ordered_actions.append(action_list[j].detach())
            actor_in = torch.cat(feats_det + state_list + ordered_actions, dim=-1)
            actor_q_i = centralized_critic(actor_in)[:, i]
            actor_loss = (agents[i].alpha.detach() * log_pi_i - actor_q_i).mean()
            writer.add_scalar('train_actor/loss_%d' % i, actor_loss, step)
            entropy = 0.5 * log_std_i.shape[1] * (1.0 + np.log(2 * np.pi)) + log_std_i.sum(dim=-1)
            writer.add_scalar('train_actor/entropy_%d' % i, entropy.mean(), step)
            agents[i].actor_optimizer.zero_grad()
            actor_loss.backward()
            agents[i].actor_optimizer.step()

            agents[i].log_alpha_optimizer.zero_grad()
            alpha_loss = (agents[i].alpha * (-log_pi_i - agents[i].target_entropy).detach()).mean()
            alpha_loss.backward()
            agents[i].log_alpha_optimizer.step()

    if step % critic_target_update_freq == 0:
        soft_update_params(centralized_critic, centralized_critic_target, critic_tau)
        # soft-update the per-agent target encoders as well (follows the
        # standard SAC-AE practice of a moving-average perception target)
        for i in range(num_agents):
            soft_update_params(agents[i].critic.encoder, agents[i].critic_target.encoder, encoder_tau)

    # also keep the actorâ encoders tied after the step (robustness)
    for i in range(num_agents):
        agents[i].actor.encoder.copy_conv_weights_from(agents[i].critic.encoder)

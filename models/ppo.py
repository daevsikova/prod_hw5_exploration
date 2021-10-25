import os

from PIL import Image
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam

class Actor(nn.Module):
    def __init__(self, opt):
        super().__init__()
        conv_layers = []
        in_c = 4
        w = opt.observation_size
        for out_c, kernel, _, s in opt.conv_filters:
            kernel = (int(kernel[1]), int(kernel[1]))
            conv_layers += [nn.Conv2d(in_c, out_c, kernel, stride=s, padding=1), nn.ReLU()]
            w = int(np.floor((w - kernel[0] + 2) / s + 1))
            in_c = out_c
        self.conv_layers = nn.Sequential(*conv_layers)
        
        linear_layers = []
        self.in_c_lin = out_c * w * w
        prev_dim = self.in_c_lin
        for dim in opt.lin_layers:
            linear_layers += [nn.Linear(prev_dim, dim), nn.ReLU()]
            prev_dim = dim
        linear_layers += [nn.Linear(prev_dim, opt.action_dim), nn.Softmax(dim=-1)]
        self.linear_layers = nn.Sequential(*linear_layers)
        
    def act(self, state):
        action_probs = self.conv_layers(state)
        action_probs = self.linear_layers(action_probs.view(-1, self.in_c_lin))
        dist = torch.distributions.Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        
        return action.detach(), action_logprob.detach()

    def compute_proba(self, state, action):
        # Returns probability of action according to current policy and distribution of actions (use it to compute entropy loss) 
        action_probs = self.conv_layers(state)
        action_probs = self.linear_layers(action_probs.view(-1, self.in_c_lin))
        dist = torch.distributions.Categorical(action_probs)
        action_logprob = dist.log_prob(action)
        dist_entropy = dist.entropy()
        
        return action_logprob, dist_entropy
        
        
class Critic(nn.Module):
    def __init__(self, opt):
        super().__init__()
        conv_layers = []
        in_c = 4
        w = opt.observation_size
        for out_c, kernel, _, s in opt.conv_filters:
            kernel = (int(kernel[1]), int(kernel[1]))
            conv_layers += [nn.Conv2d(in_c, out_c, kernel, stride=s, padding=1), nn.ReLU()]
            w = int(np.floor((w - kernel[0] + 2) / s + 1))
            in_c = out_c
        self.conv_layers = nn.Sequential(*conv_layers)
        
        linear_layers = []
        self.in_c_lin = out_c * w * w
        prev_dim = self.in_c_lin
        for dim in opt.lin_layers:
            linear_layers += [nn.Linear(prev_dim, dim), nn.ReLU()]
            prev_dim = dim
        linear_layers += [nn.Linear(prev_dim, opt.out_dim_critic)]
        self.linear_layers = nn.Sequential(*linear_layers)
        
    def get_value(self, state):
        return self.linear_layers(self.conv_layers(state).view(-1, self.in_c_lin))


class PPO:
    def __init__(self, opt):
        self.opt = opt
        self.device = self.opt.device
        self.actor = Actor(self.opt).to(self.device)
        self.critic = Critic(self.opt).to(self.device)
        self.actor_optim = Adam(self.actor.parameters(), self.opt.actor_lr)
        self.critic_optim = Adam(self.critic.parameters(), self.opt.critic_lr)

    def update(self, trajectories):
        transitions = [t for traj in trajectories for t in traj] # Turn a list of trajectories into list of transitions
        state, action, old_log_prob, target_value, advantage= zip(*transitions)
        state = torch.cat(state, dim=0).to(self.device)
        action = torch.Tensor(action).to(self.device)
        old_log_prob = torch.Tensor(old_log_prob).to(self.device)
        target_value = torch.Tensor(target_value).to(self.device)
        advantage = torch.Tensor(advantage).to(self.device)
        target_value = (target_value - target_value.mean()) / (target_value.std() + 1e-8)
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        self._loss_actor = 0.0
        self._loss_critic = 0.0
        
        for _ in range(self.opt.batches_per_update):
            idx = np.random.randint(0, len(transitions), self.opt.batch_size) # Choose random batch
            s = state[idx].float()
            a = action[idx].float()
            log_old_prob = old_log_prob[idx].float() # Probability of the action in state s.t. old policy
            tar_val = target_value[idx].float() # Estimated by lambda-returns 
            adv = advantage[idx].float()
            
            log_prob, distr_entropy = self.actor.compute_proba(s, a)

            ratios = torch.exp(log_prob - log_old_prob.detach())
            surr1 = ratios * adv
            surr2 = torch.clamp(ratios, 1.0 - self.opt.clip, 1.0 + self.opt.clip) * adv
            distr_entropy = distr_entropy.mean()
            self.entropy = distr_entropy.item()
            actor_loss = -torch.min(surr1, surr2).mean() - self.opt.entropy_coef * distr_entropy

            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()

            self._loss_actor += actor_loss.item()

            value_pred = self.critic.get_value(s)
            critic_loss = F.mse_loss(value_pred, tar_val)
            
            self.critic_optim.zero_grad()
            critic_loss.backward()
            self.critic_optim.step()

            self._loss_critic += critic_loss.item()

        self._loss_actor /= self.opt.batches_per_update
        self._loss_critic /= self.opt.batches_per_update
            
    def get_value(self, state):
        value = self.critic.get_value(state)
        return value

    def act(self, state):
        action, action_logprob = self.actor.act(state)
        return action, action_logprob

    def save_networks(self, iter):
        os.makedirs(self.opt.save_dir, exist_ok=True)
        paths = []
        for name in ['actor', 'critic']:
            save_filename = '%s_net_%s.pth' % (iter, name)
            save_path = os.path.join(self.opt.save_dir, save_filename)
            net = getattr(self, name)

            if torch.cuda.is_available():
                torch.save(net.module.cpu().state_dict(), save_path)
            else:
                torch.save(net.cpu().state_dict(), save_path)
            paths.append(save_path)
        return paths


def evaluate_policy(env, agent):
    rewards = []
    for i in range(5):
        obs = env.reset()
        n = 400
        frames = []
        cur_reward = 0.0
        
        if i == 4:
            Image.fromarray(env._map.render(env._agent))\
                .convert('RGB')\
                .resize((500, 500), Image.NEAREST)\
                .save(f'{agent.opt.save_dir}/tmp.png')

        for _ in range(n):
            obs = torch.tensor(obs).float().permute(2, 0, 1).unsqueeze(0)
            with torch.no_grad():
                action, _ = agent.act(obs)
            
            if i == 4:
                frame = Image.fromarray(env._map.render(env._agent)).convert('RGB').resize((500, 500), Image.NEAREST).quantize()
                frames.append(frame)

            obs, reward, done, _ = env.step(action.item())
            cur_reward += reward
            if done:
                break 

        if i == 4:
            frames[0].save(f"{agent.opt.save_dir}/out.gif", save_all=True, append_images=frames[1:], loop=0, duration=1000/60)

        rewards.append(cur_reward)

    return np.array(rewards)
   

def sample_episode(env, agent):
    obs = env.reset()
    done = False
    trajectory = []
    while not done:
        obs = torch.tensor(obs).float().permute(2, 0, 1).unsqueeze(0).to(agent.device)
        action, action_logprob = agent.act(obs)
        value = agent.get_value(obs)
        n_obs, reward, done, _ = env.step(action.item())
        trajectory.append((obs, action, reward, action_logprob, value))
        obs = n_obs
    return compute_lambda_returns_and_gae(trajectory, agent.opt)


def compute_lambda_returns_and_gae(trajectory, opt):
    lambda_returns = []
    gae = []
    last_lr = 0.
    last_value = 0.
    for _, _, reward, _, value in reversed(trajectory):
        # ret = reward + (opt.gamma * last_lr)
        ret = reward + opt.gamma * (last_value * (1 - opt.lambda_) + last_lr * opt.lambda_)
        last_lr = ret
        last_value = value
        lambda_returns.append(last_lr)
        gae.append(last_lr - value)
    
    # Each transition contains state, action, old action probability, value estimation and advantage estimation
    return [(obs, act, prob, val, adv) for (obs, act, _, prob, _), val, adv in zip(trajectory, reversed(lambda_returns), reversed(gae))]
    # return [(obs, act, prob, val) for (obs, act, _, prob, _), val in zip(trajectory, reversed(lambda_returns))]

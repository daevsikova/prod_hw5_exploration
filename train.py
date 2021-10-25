from omegaconf import OmegaConf
from argparse import Namespace
import wandb
import torch
from datetime import datetime
import os
import sys

from gym import spaces

sys.path.append(os.path.abspath("mapgen"))
os.environ["PYTHONPATH"] = os.path.abspath("mapgen")
from mapgen import Dungeon
from utils.options import BaseOptions
from models.ppo import PPO, sample_episode, evaluate_policy



class ModifiedDungeon(Dungeon):
    """Use this class to change the behavior of the original env (e.g. remove the trajectory from observation, like here)"""
    def __init__(self, opt):
        super().__init__(
            width=opt.width,
            height=opt.height,
            max_rooms=opt.max_rooms,
            min_room_xy=opt.min_room_xy,
            max_room_xy=opt.max_room_xy,
            observation_size = opt.observation_size,
            vision_radius = opt.vision_radius,
            max_steps = opt.max_steps
        )

        self.observation_space = spaces.Box(0, 1, [opt.observation_size, opt.observation_size, 3]) # because we remove trajectory and leave only cell types (UNK, FREE, OCCUPIED)
        self.action_space = spaces.Discrete(3)
        self.reward_type = opt.reward

    def step(self, action):
        obs, reward, done, info = super().step(action)
        if self.reward_type == 'custom':
            reward = self.custom_reward(info)
        return obs, reward, done, info
    
    def custom_reward(self, info):
        moved = info['moved'] 
        new_cells = info['new_explored']
        ratio_explored = info['total_explored']  / info['total_cells']
        reward = - 1
        if moved:
            reward = 0.1 + ratio_explored * 20 if new_cells > 0 else -0.5
        return reward

def fix_seed(seed=21):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

if __name__ == "__main__":
    fix_seed()

    opt = BaseOptions().parse()  # get training options
    if vars(opt).get("config", None) is not None:
        cfg = OmegaConf.load(opt.config)
        opt = OmegaConf.merge(vars(opt), cfg)
        opt = Namespace(**opt)

    opt.device = 'cuda' if opt.device =='cuda' and torch.cuda.is_available() else 'cpu'
    opt.save_dir = os.path.join(opt.ckpt_dir, opt.name)
    os.makedirs(opt.ckpt_dir, exist_ok=True)
    os.makedirs(opt.save_dir, exist_ok=True)
    
    env = ModifiedDungeon(opt)
    ppo = PPO(opt=opt)
    state = env.reset()
    episodes_sampled = 0
    steps_sampled = 0
    
    if not os.environ.get("WANDB_API_KEY", None):
        os.environ["WANDB_API_KEY"] = "e891f26c3ad7fd5a7e215dc4e344acc89c8861da"
    name = opt.name + datetime.strftime(datetime.now(), "_%h%d_%H%M%S")
    wandb.init(project='prod_hw5_rl', entity='daevsikova', config=opt, name=name)
    
    for i in range(opt.n_iter):
        trajectories = []
        steps_cnt = 0
        
        while len(trajectories) < opt.min_episodes_per_update or steps_cnt < opt.min_transitions_per_update:
            trajectory = sample_episode(env, ppo)
            steps_cnt += len(trajectory)
            trajectories.append(trajectory)
        episodes_sampled += len(trajectories)
        steps_sampled += steps_cnt

        ppo.update(trajectories)
        wandb.log({'Critic loss': ppo._loss_critic, 'Actor loss': ppo._loss_actor, 'Entropy': ppo.entropy})
        
        if (i + 1) % opt.eval_freq == 0:
            reward = evaluate_policy(env, ppo)
            wandb.log({"Trajectory": wandb.Video(f"{opt.save_dir}/out.gif", fps=15, format="gif")})
            wandb.log({'Reward': reward.mean()})
            print(f"Iteration: {i + 1}, Reward mean: {reward.mean()}, Reward std: {reward.std()}, Episodes sampled: {episodes_sampled}")
        if (i + 1) % opt.save_freq == 0: 
            paths = ppo.save_networks(i + 1)
            for p in paths:
                wandb.save(p)

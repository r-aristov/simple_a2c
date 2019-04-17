import torch
import torch.nn.functional as F
import torch
from torch import nn as nn, optim as optim
from torch.distributions import Categorical
import gym
from train_batch import Brain, get_replay


def main():
    agent = Brain(4,2)
    agent.load("trained.nn")
    env = gym.make('CartPole-v0')
    while True:
        get_replay(agent, env, True)

if __name__ == '__main__':
    main()
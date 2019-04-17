import torch
import torch.nn.functional as F
import torch
from torch import nn as nn, optim as optim
from torch.distributions import Categorical
import gym


class Replay:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.discounted_rewards = []
        self.iter = 0
        self.total_reward = 0.0

class A2CAgent:
    def __init__(self):
        self.__iter = 0
        self.replay = Replay()
        self.total_reward = 0.0

    def decide(self, x):
        self.replay.states.append(x)
        probs, value = self.forward(x)

        distrib = Categorical(probs)
        action = distrib.sample().item()
        self.replay.actions.append(action)
        self.__iter += 1
        self.replay.iter = self.__iter
        return action

    def reward(self, r):
        self.replay.rewards.append(r)
        self.total_reward += r

    def forward(self, x):
        raise NotImplementedError

    def end_replay(self):
        replay = self.replay
        replay.iter = self.__iter
        replay.total_reward = self.total_reward
        self.replay = Replay()
        self.__iter = 0
        self.total_reward = 0.0
        return replay


class Brain(nn.Module, A2CAgent):
    def __init__(self, state_features_count, action_count, hidden_size=64):
        nn.Module.__init__(self)
        A2CAgent.__init__(self)

        self.affine1 = nn.Linear(state_features_count, hidden_size)
        self.action_head = nn.Linear(hidden_size, action_count)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h = F.relu(self.affine1(x))
        action_scores = self.action_head(h)
        value = self.value_head(h)
        return F.softmax(action_scores, dim=-1), value

    def save(self, filename):
        f = open(filename, "wb")
        torch.save(self.state_dict(), f)
        f.close()

    def load(self, filename):
        state_dict = torch.load(filename)
        self.load_state_dict(state_dict)


def discount_rewards(rewards, gamma):
    values = []
    v = 0.0

    for r in rewards[::-1]:
        v = r + gamma * v
        values.insert(0, torch.tensor(v))

    return values


def a2c_loss(logprobs, pred_vals, true_vals):
    l = logprobs.shape[0]
    advantage = (true_vals - pred_vals).detach()
    ploss = -(logprobs * advantage)
    vloss = F.smooth_l1_loss(pred_vals, true_vals, reduce=False)
    loss = (ploss + vloss)
    loss = loss.sum()
    return loss / l


def get_replay(agent, env, render=False):
    state = torch.tensor(env.reset()).float()
    done = False

    while not done:
        action = agent.decide(state)
        if render:
            env.render()
        new_state, reward, done, _ = env.step(action)
        state = torch.tensor(new_state).float()
        agent.reward(reward)
    replay = agent.end_replay()
    return replay


def get_loss(agent, replays):
    logprob_list = []
    pred_values_list = []
    replay = replays[0]
    for i in range(len(replay.states)):
        state = replay.states[i]
        action = torch.tensor(replay.actions[i])

        prob, pred_v = agent.forward(state)
        distrib = Categorical(prob)
        logprob = distrib.log_prob(action)
        logprob_list.append(logprob)
        pred_values_list.append(pred_v)

    logprobs = torch.stack(logprob_list).squeeze_()
    pred_vals = torch.stack(pred_values_list).squeeze_()
    true_vals = torch.stack(replay.discounted_rewards).squeeze_()
    loss = a2c_loss(logprobs, pred_vals, true_vals)
    return loss


def main():
    env = gym.make('CartPole-v0')
    agent = Brain(4, 2)
    optimizer = optim.Adam(agent.parameters(), lr=1e-2)
	
    replays = []
    replays_in_batch = 1
    smoothed_reward_list = []
    filter_len = 10
    display_iter = 20
    max_reward = 196
    render = True

    for episode in range(100000):

        replay = get_replay(agent, env, render)
        render = False

        if len(smoothed_reward_list) > filter_len:
            del smoothed_reward_list[0]
        smoothed_reward_list.append(replay.total_reward)
        smoothed_reward = sum(smoothed_reward_list) / len(smoothed_reward_list)

        if (episode % display_iter == 0) or (smoothed_reward > max_reward):
            print("#%d Simulation complete in %d steps, reward: %1.3f, smoothed reward %1.3f" % (episode, replay.iter, replay.total_reward, smoothed_reward))
            render = True

        if smoothed_reward > max_reward:
            print("Training done in %d iterations" % episode)
            agent.save("trained.nn")
            break

        replay.discounted_rewards = discount_rewards(replay.rewards, 0.999)
        replays.append(replay)

        if len(replays) >= replays_in_batch:
            loss = get_loss(agent, replays)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            replays = []


if __name__ == '__main__':
    main()

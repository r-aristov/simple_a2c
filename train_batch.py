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
        self.replay.actions.append(torch.tensor(action).unsqueeze_(0))
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
        values.insert(0, torch.tensor(v).unsqueeze_(0))

    return values


def a2c_loss(logprobs, pred_vals, true_vals, mask):
    advantage = (true_vals - pred_vals).detach()
    ploss = -(logprobs * advantage)
    vloss = F.smooth_l1_loss(pred_vals, true_vals, reduce=False)
    loss = (ploss + vloss) * mask
    loss = loss.sum() / mask.sum()
    return loss


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


def get_padded_tensor(tensor_list, max_len):
    seq_len = len(tensor_list)
    stacked = torch.stack(tensor_list)
    stacked = F.pad(stacked, (0,0,0, max_len - seq_len))
    stacked.unsqueeze_(0)
    return stacked


def get_loss(agent, replays):
    max_len = len(max(replays, key=lambda r: len(r.rewards)).rewards)
    batch_size = len(replays)
    mask = torch.ones(batch_size, max_len)

    states = get_padded_tensor(replays[0].states, max_len)
    actions = get_padded_tensor(replays[0].actions, max_len)
    true_rewards = get_padded_tensor(replays[0].discounted_rewards, max_len)
    i = 0

    for r in replays:

        state = get_padded_tensor(r.states, max_len)
        action = get_padded_tensor(r.actions, max_len)
        reward = get_padded_tensor(r.discounted_rewards, max_len)

        states = state if i == 0 else torch.cat((states, state))
        actions = action if i == 0 else torch.cat((actions, action))
        true_rewards = reward if i == 0 else torch.cat((true_rewards, reward))

        mask[i, range(len(r.states), max_len)] = 0.0
        i += 1

    states = torch.split(states, 1, dim=1)
    actions = torch.split(actions, 1, dim=1)

    logprob_list = []
    pred_values_list = []

    for s, a in zip(states, actions):
        s.squeeze_()
        a.squeeze_()

        prob, pred_v = agent.forward(s)

        distrib = Categorical(prob)
        logprob = distrib.log_prob(a)

        logprob_list.append(logprob)
        pred_values_list.append(pred_v)

    logprobs = torch.stack(logprob_list).t()

    pred_vals = torch.stack(pred_values_list).squeeze_().t()

    true_rewards.squeeze_()

    loss = a2c_loss(logprobs, pred_vals, true_rewards, mask)
    return loss


def main():
    env = gym.make('CartPole-v0')
    agent = Brain(4, 2)
    optimizer = optim.Adam(agent.parameters(), lr=1e-2)
    replays = []
    replays_in_batch = 4
    smoothed_reward_list = []
    filter_len = 10
    display_iter = 20
    max_reward = 196
    batch = 0
    render = True

    for episode in range(100000):

        replay = get_replay(agent, env, render)
        render = False

        replay.discounted_rewards = discount_rewards(replay.rewards, 0.999)
        replays.append(replay)

        if len(replays) >= replays_in_batch:
            loss = get_loss(agent, replays)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            replays = []

            if len(smoothed_reward_list) > filter_len:
                del smoothed_reward_list[0]
            smoothed_reward_list.append(replay.total_reward)
            smoothed_reward = sum(smoothed_reward_list) / len(smoothed_reward_list)

            if (batch % display_iter == 0) or (smoothed_reward > max_reward):
                print("#%d Simulation complete in %d steps, reward: %1.3f, smoothed reward %1.3f" % (batch, replay.iter, replay.total_reward, smoothed_reward))
                render = True

            if smoothed_reward > max_reward:
                print("Training done in %d iterations" % batch)
                agent.save("trained.nn")
                break

            batch += 1


if __name__ == '__main__':
    main()

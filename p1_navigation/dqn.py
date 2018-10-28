from unityagents import UnityEnvironment
import numpy as np
import torch
import torch.nn as nn
import tensorflow as tf
import collections
from collections import deque
import torchvision
from torchvision import transforms
import torch.nn.functional as F
import torch.optim as optim
import random

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64  # minibatch size
GAMMA = 0.99  # discount factor
TAU = 1e-3  # for soft update of target parameters
LR = 5e-4  # learning rate
UPDATE_EVERY = 4  # how often to update the network
train_steps_per_episode = 100
eps_start = 0.5
eps_decay = 0.999
eps_end = 0.1
tn = 1
tau = 1e-4
Transition = collections.namedtuple('Transition',
                                    'state action reward next_state done')
wabs_default = 0.1


def sample_data(memory, weights, n):
    wabs = np.abs(weights)
    wabs = wabs / np.sum(wabs)
    indexes = np.random.choice(len(memory), n, p=wabs)
    return indexes, [memory[i] for i in indexes]


def update_weights(weights, indexes, new_weights):
    wabs_default = max(abs(new_weights))
    for i, ind in zip(range(len(indexes)), indexes):
        weights[ind] = new_weights[i]


def todict(l):
    return dict(zip(range(len(l)), l))


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.01)


def create_qfn(input_size=37,
               output_size=4,
               intermediate_layers=[30, 20, 20, 20, 10],
               tn=1):
    lsizes = [input_size * tn] + intermediate_layers + [output_size]
    ldims = list(zip(lsizes[:-1], lsizes[1:]))
    layers = []
    for idim, odim in ldims[:-1]:
        layers.append(nn.Linear(idim, odim))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(*ldims[-1]))
    net = nn.Sequential(*layers)
    net.apply(init_weights)
    return net

def main(env, state_size, action_size, num_time_steps_per_state):
    qfn_local = create_qfn(
        input_size=state_size,
        output_size=action_size,
        tn=num_time_steps_per_state).to(device)
    qfn_target = create_qfn(
        input_size=state_size,
        output_size=action_size,
        tn=num_time_steps_per_state).to(device)
    optimizer = optim.Adam(qfn_local.parameters(), lr=LR)
    memory = deque(maxlen=BUFFER_SIZE)
    weights = deque(maxlen=BUFFER_SIZE)
    scores = []
    scores_window = deque(maxlen=100)
    eps = eps_start
    save = 0
    for i in range(10):
        save = (save + 1) % 100
        scores.append(
            episode(env, epsilon_greedy_policy(qfn_local, eps), memory,
                    weights))
        scores_window.append(scores[-1])
        for train_step in range(train_steps_per_episode):
            indices, samples = sample_data(memory, weights, BATCH_SIZE)
            dq = update_params(qfn_local, qfn_target, optimizer, samples,
                               GAMMA)
            #print([(e.reward,w) for e,w in zip(samples,list(dq.squeeze().numpy())) if abs(e.reward)>0.1])
            update_weights(weights, indices, dq)
        eps = max(eps_end, eps * eps_decay)
        print('eps : ', eps)
        if save == 0:
            torch.save(qfn_local.state_dict(), 'checkpoint.pth')
        if i > 100:
            print(i - 100, np.mean(scores_window))


def samples_to_tensors(samples):
    states = torch.from_numpy(
        np.vstack([np.hstack(e.state) for e in samples])).float().to(device)
    actions = torch.from_numpy(np.vstack(
        [e.action for e in samples])).long().to(device)
    rewards = torch.from_numpy(np.vstack(
        [e.reward for e in samples])).float().to(device)
    next_states = torch.from_numpy(
        np.vstack([np.hstack(e.state) for e in samples])).float().to(device)
    dones = torch.from_numpy(
        np.vstack([e.done for e in samples]).astype(
            np.uint8)).float().to(device)
    return states, actions, rewards, next_states, dones


def update_params(qfn_local, qfn_target, optimizer, samples, gamma):
    states, actions, rewards, next_states, dones = samples_to_tensors(samples)
    q_next_targets = qfn_target(next_states).detach().max(1)[0].unsqueeze(
        1)  # modify to try double-dqn
    q_targets = rewards + gamma * q_next_targets * (1 - dones)
    q_expected = qfn_local(states).gather(1, actions)
    dq = (q_expected.detach() - q_targets.detach()).detach()
    loss = F.mse_loss(q_expected, q_targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()  # this updates local network
    soft_update_target(qfn_target,
                       qfn_local)  # note target network is being updated
    return dq


def soft_update_target(qfn_target, qfn_local):
    for target_param, local_param in zip(qfn_target.parameters(),
                                         qfn_local.parameters()):
        target_param.data.copy_(tau * local_param.data +
                                (1.0 - tau) * target_param.data)


# qfn(state)->{action:float_val}
def epsilon_greedy_policy(qfn, eps):
    greed = 1.0 - eps

    def policy(state):
        qfn.eval()
        with torch.no_grad():
            possible_actions = qfn(
                torch.from_numpy(
                    np.hstack(state)).float().to(device).unsqueeze(0))
        qfn.train()
        num_actions = len(possible_actions)
        p = eps / num_actions
        if not isinstance(possible_actions, dict):
            possible_actions = todict(possible_actions)
        greedy_action = max(possible_actions, key=possible_actions.get)
        probs = [
            p + greed if i == greedy_action else p for i in range(num_actions)
        ]
        return np.random.choice(range(num_actions), p=probs)

    return policy


def episode(env, policy, memory, weights):
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=False)[brain_name]  # reset the environment
    state = env_info.vector_observations[0]  # get the current state
    states = collections.deque(maxlen=tn)
    for i in range(tn):
        states.append(state)
    score = 0
    num_actions = 0
    rewards = []
    while True:
        action = policy(states)  # select an action
        env_info = env.step(action)[
            brain_name]  # send the action to the environment
        next_state = env_info.vector_observations[0]  # get the next state
        reward = env_info.rewards[0]  # get the reward
        score += reward
        rewards.append(reward)
        num_actions += 1
        pstate = list(states)
        states.append(next_state)
        nstate = list(states)
        done = env_info.local_done[0]  # see if episode has finished
        if len(pstate) == tn:
            memory.append(Transition(pstate, action, reward, nstate, done))
            weights.append(wabs_default * (abs(reward) + 0.1))
        if done:  # exit loop if episode finished
            print('episode length : ', num_actions, ' score : ', score)
            rewards_out = {
                i: r
                for i, r in zip(range(num_actions), rewards) if r != 0
            }
            print(rewards_out)
            break
    return score


if __name__ == '__main__':
    mac_path = "Banana.app"
    linux_path = "Banana_Linux/Banana.x86_64"
    linux_headless_path = "Banana_Linux_NoVis/Banana.x86_64"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    env = UnityEnvironment(file_name=linux_headless_path)
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=True)[brain_name]
    main(env, 37, 4, 1)

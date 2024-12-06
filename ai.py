import math
import random
from collections import namedtuple, deque
from itertools import count
import pickle
import time
import signal
import os
import tetris
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pygame

pygame.init()

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


NEURON_AMOUNT = 32

class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, NEURON_AMOUNT)
        self.layer2 = nn.Linear(NEURON_AMOUNT, NEURON_AMOUNT)
        self.layer3 = nn.Linear(NEURON_AMOUNT, NEURON_AMOUNT)
        self.layer4 = nn.Linear(NEURON_AMOUNT, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        return self.layer4(x)


BATCH_SIZE = 512
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.2
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

n_actions = tetris.n_actions
n_observations = tetris.n_observations

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=False)
memory = ReplayMemory(10000)

steps_done = 0

save_to_file = False

def sigint_handler(signal, frame):
    global save_to_file
    save_to_file = True

signal.signal(signalnum=signal.SIGINT, handler=sigint_handler)

started = 0
control = True

def select_action(state):
    global started
    global control
    started += 1
    if started > 500 and control:
        while True and control:
            ret = None
        
            events = pygame.event.get()
            for event in events:
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        ret = 1
                    elif event.key == pygame.K_RIGHT:
                        ret = 2
                    elif event.key == pygame.K_UP:
                        ret = 3
                    elif event.key == pygame.K_r:
                        global save_to_file
                        save_to_file = True
                    elif event.key == pygame.K_g:
                        control = False
                    
            keys = pygame.key.get_pressed()
            if keys[pygame.K_DOWN]:
                ret = 0

            if ret != None:
                return torch.tensor([[ret]], device=device, dtype=torch.long)

    events = pygame.event.get()
    for event in events:
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_g:
                control = True
            elif event.key == pygame.K_d:
                        tetris.change_display()

    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1

    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.

            # TODO select first allowed move

            return policy_net(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[random.randint(0, n_actions-1)]], device=device, dtype=torch.long)

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)

    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

counter = 0

def train_once():
    tetris.reset()
    state = torch.tensor(tetris.get_state(), dtype=torch.float32, device = torch.device(
                            "cuda" if torch.cuda.is_available() else
                            "mps" if torch.backends.mps.is_available() else
                            "cpu"
)).unsqueeze(0)
    print("stsr")
    for t in count():
        action = select_action(state)
        observation, reward, terminated = tetris.step(action.item())
        reward = torch.tensor([reward], device=device)

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        memory.push(state, action, next_state, reward)

        state = next_state

        global counter
        if counter % 100 == 0:
            optimize_model()
        counter += 1

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        if save_to_file:
            save_model()
            exit()

        if terminated:
            break

def train_model(num_episodes):
    #for i_episode in range(num_episodes):
    while True:
        train_once()

def save_model():
    with open(f"saved_models/model_{time.time()}", "wb") as f:
        pickle.dump({"policy_net": policy_net, "memory": memory, "steps_done": steps_done}, f)

def load_model(file, keep = True):
    with open(file, "rb") as f:
        global steps_done, memory
        t = pickle.load(f)
        temp_net = t["policy_net"]
        memory = t["memory"]
        if keep:
            steps_done = t["steps_done"]

        """ temp_net = pickle.load(f) """

        global policy_net, target_net
        policy_net.load_state_dict(temp_net.state_dict())
        target_net.load_state_dict(policy_net.state_dict())


if torch.cuda.is_available() or torch.backends.mps.is_available():
    num_episodes = 600
else:
    num_episodes = 50

if len(sys.argv) == 2:
    load_model(sys.argv[1])
elif len(sys.argv) == 3:
        load_model(sys.argv[1], sys.argv[2])
        


train_model(num_episodes)
save_model()

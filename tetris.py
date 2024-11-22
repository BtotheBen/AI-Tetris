import torch
import main

n_actions = 4 # TODO
n_observations = 204 # TODO

# reset the game and return the base state
def reset():
    global m
    m = main.Main()

def get_state():
    return m.get_state()

# execute one step, returns tuple (observation, reward, terminated)
def step(action):
    reward, terminated = m.run(action)
    return get_state(), reward, terminated


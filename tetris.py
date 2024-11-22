import torch
import main
import time

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
    #time.sleep(0.100)
    reward, terminated = m.run(action)
    return get_state(), reward, terminated


import torch
import gamecode.main as main

n_actions: int = 5  # number of actions that can be taken each frame
n_observatins: int

# reset the game and return the base state
def reset() -> torch.Tensor:
    global m 
    m = main.Main()
    
def get_state():
    return m.get_state()

# execute one step, returns tuple (observation, reward, terminated)
def step(action):
    m.run(action)
    m.get_state()


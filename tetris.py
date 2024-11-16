import torch
import main

class tetrisenv():
    def __init__(self) -> None:
        self.n_actions: int = 5  # number of actions that can be taken each frame
        self.n_observations: int = 5

    # reset the game and return the base state
    def reset(self) -> torch.Tensor:
        global m 
        m = main.Main()
        m.start()
        
    def get_state(self):
        return m.get_state()

    # execute one step, returns tuple (observation, reward, terminated)
    def step(self, action):
        m.run(action)
        m.get_state()


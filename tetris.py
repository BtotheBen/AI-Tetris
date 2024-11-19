import torch
import main

class tetrisenv():
    def __init__(self) -> None:
        self.n_actions: int = 5  # number of actions that can be taken each frame
        self.n_observations: int = 204

    # reset the game and return the base state
    def reset(self) -> torch.Tensor:
        global m 
        m = main.Main()
        m.start()
        print(m.get_state())
        return self.get_state()
        
    def get_state(self):
        return torch.tensor((torch.tensor(m.get_state())), dtype=torch.float32, device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)).unsqueeze(0)

    # execute one step, returns tuple (observation, reward, terminated)
    def step(self, action):
        reward = m.run(action)
        return self.get_state(), reward


import torch
import main
import time

n_actions = 4 # TODO
n_observations = 204 # TODO

display_draw = True

# reset the game and return the base state
def reset():
    global m
    global display_draw
    m = main.Main()
    m.draw = display_draw

def get_state():
    return m.get_state()

# execute one step, returns tuple (observation, reward, terminated)
def step(action):
    #time.sleep(0.100)
    reward, terminated = m.run(action)
    return get_state(), reward, terminated

def change_display():
    global display_draw
    m.draw = not m.draw
    display_draw = not display_draw
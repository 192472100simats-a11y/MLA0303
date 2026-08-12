# Q6: RL model for autonomous robot navigation
# Uses a Gymnasium-compatible custom environment and a Keras neural network.
# If TensorFlow/Gymnasium are unavailable, install:
# pip install tensorflow gymnasium numpy

import numpy as np
import random
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

GRID = 5
GOAL = (4, 4)
OBSTACLES = {(1, 2), (2, 2), (3, 1)}

ACTIONS = [(-1,0), (1,0), (0,-1), (0,1)]

def step(state, action):
    r, c = state
    dr, dc = ACTIONS[action]
    nr, nc = r + dr, c + dc

    if nr < 0 or nr >= GRID or nc < 0 or nc >= GRID:
        return state, -5, False
    if (nr, nc) in OBSTACLES:
        return state, -5, False
    if (nr, nc) == GOAL:
        return (nr, nc), 20, True
    return (nr, nc), -1, False

def encode(state):
    x = np.zeros(GRID * GRID, dtype=np.float32)
    x[state[0] * GRID + state[1]] = 1
    return x

model = Sequential([
    Dense(64, activation="relu", input_shape=(GRID * GRID,)),
    Dense(64, activation="relu"),
    Dense(4, activation="linear")
])
model.compile(optimizer="adam", loss="mse")

gamma = 0.95
epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 0.995

for episode in range(500):
    state = (0, 0)

    for _ in range(100):
        if random.random() < epsilon:
            action = random.randrange(4)
        else:
            q = model.predict(encode(state)[None], verbose=0)[0]
            action = int(np.argmax(q))

        next_state, reward, done = step(state, action)

        target = model.predict(encode(state)[None], verbose=0)[0]
        if done:
            target[action] = reward
        else:
            next_q = model.predict(encode(next_state)[None], verbose=0)[0]
            target[action] = reward + gamma * np.max(next_q)

        model.fit(encode(state)[None], target[None], epochs=1, verbose=0)

        state = next_state
        if done:
            break

    epsilon = max(epsilon_min, epsilon * epsilon_decay)

# Evaluation
state = (0, 0)
path = [state]

for _ in range(30):
    q = model.predict(encode(state)[None], verbose=0)[0]
    action = int(np.argmax(q))
    state, _, done = step(state, action)
    path.append(state)
    if done:
        break

print("Q6 - Deep RL Robot Navigation")
print("Learned path:", path)
print("Reached goal:", state == GOAL)

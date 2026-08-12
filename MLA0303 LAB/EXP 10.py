# Q10: Deep Q-Network for autonomous drone delivery
# State = (location, battery level)
# Action = move North, South, East, West
# Battery decreases after each movement.
#
# Requires:
# pip install tensorflow numpy

import random
import numpy as np
import tensorflow as tf
from collections import deque
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

GRID = 5
MAX_BATTERY = 20
START = (0, 0)
GOAL = (4, 4)

ACTIONS = [(-1,0), (1,0), (0,-1), (0,1)]

def encode(state):
    r, c, battery = state
    x = np.zeros(GRID * GRID + MAX_BATTERY + 1, dtype=np.float32)
    x[r * GRID + c] = 1
    x[GRID * GRID + battery] = 1
    return x

def step(state, action):
    r, c, battery = state

    if battery <= 0:
        return state, -20, True

    dr, dc = ACTIONS[action]
    nr, nc = r + dr, c + dc
    nb = battery - 1

    if not (0 <= nr < GRID and 0 <= nc < GRID):
        return state, -5, False

    if (nr, nc) == GOAL:
        return (nr, nc, nb), 50, True

    # Battery penalty encourages efficient routes
    return (nr, nc, nb), -1, False

input_size = GRID * GRID + MAX_BATTERY + 1

model = Sequential([
    Dense(128, activation="relu", input_shape=(input_size,)),
    Dense(128, activation="relu"),
    Dense(4, activation="linear")
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="mse"
)

memory = deque(maxlen=5000)

gamma = 0.95
epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 0.995

batch_size = 32

for episode in range(1000):
    state = (0, 0, MAX_BATTERY)

    for _ in range(50):
        if random.random() < epsilon:
            action = random.randrange(4)
        else:
            q = model.predict(encode(state)[None], verbose=0)[0]
            action = int(np.argmax(q))

        next_state, reward, done = step(state, action)
        memory.append((state, action, reward, next_state, done))
        state = next_state

        if len(memory) >= batch_size:
            batch = random.sample(memory, batch_size)

            states = np.array([encode(x[0]) for x in batch])
            next_states = np.array([encode(x[3]) for x in batch])

            q_values = model.predict(states, verbose=0)
            next_q = model.predict(next_states, verbose=0)

            for i, (_, action, reward, _, done) in enumerate(batch):
                target = reward
                if not done:
                    target += gamma * np.max(next_q[i])
                q_values[i, action] = target

            model.fit(states, q_values, epochs=1, verbose=0)

        if done:
            break

    epsilon = max(epsilon_min, epsilon * epsilon_decay)

# Test trained DQN
state = (0, 0, MAX_BATTERY)
path = [state]

for _ in range(30):
    q = model.predict(encode(state)[None], verbose=0)[0]
    action = int(np.argmax(q))
    state, reward, done = step(state, action)
    path.append(state)

    if done:
        break

print("Q10 - DQN Drone Delivery")
print("Learned route:")
for p in path:
    print(p)

print("\nDelivered successfully:", state[:2] == GOAL)
print("Remaining battery:", state[2])

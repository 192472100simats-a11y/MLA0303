# Q11: Compare DQN, Double DQN, Dueling DQN and PER
# Smart traffic signal control.
# State: queue length on 4 approaches.
# Actions: choose one of 2 traffic phases.
# Reward: negative total waiting vehicles.

import random
import numpy as np
import tensorflow as tf
from collections import deque
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input, Lambda, Add, Subtract
from tensorflow.keras.optimizers import Adam

STATE_SIZE = 4
ACTIONS = 2

def environment_step(state, action):
    state = np.array(state, dtype=np.float32)
    # Vehicles arrive randomly.
    arrivals = np.random.randint(0, 3, STATE_SIZE)

    # Selected phase serves two lanes.
    served = np.zeros(STATE_SIZE)
    if action == 0:
        served[[0, 1]] = np.random.randint(1, 4, 2)
    else:
        served[[2, 3]] = np.random.randint(1, 4, 2)

    next_state = np.maximum(0, state + arrivals - served)
    reward = -float(np.sum(next_state))  # minimize waiting
    return next_state, reward

def make_dqn():
    return Sequential([
        Input(shape=(STATE_SIZE,)),
        Dense(64, activation="relu"),
        Dense(64, activation="relu"),
        Dense(ACTIONS)
    ])

def make_dueling():
    inputs = tf.keras.Input(shape=(STATE_SIZE,))
    x = Dense(64, activation="relu")(inputs)
    x = Dense(64, activation="relu")(x)

    value = Dense(1)(x)
    advantage = Dense(ACTIONS)(x)
    advantage_mean = Lambda(lambda z: tf.reduce_mean(z, axis=1, keepdims=True))(advantage)
    q = Add()([value, Subtract()([advantage, advantage_mean])])

    return tf.keras.Model(inputs, q)

def train(algorithm, episodes=300):
    online = make_dueling() if algorithm == "Dueling DQN" else make_dqn()
    target = make_dueling() if algorithm == "Dueling DQN" else make_dqn()

    online.compile(optimizer=Adam(0.001), loss="mse")
    target.set_weights(online.get_weights())

    memory = deque(maxlen=3000)
    gamma, epsilon = 0.95, 1.0
    rewards = []

    for episode in range(episodes):
        state = np.random.randint(0, 5, STATE_SIZE).astype(np.float32)
        total = 0

        for _ in range(40):
            if random.random() < epsilon:
                action = random.randrange(ACTIONS)
            else:
                action = int(np.argmax(online.predict(state[None], verbose=0)[0]))

            next_state, reward = environment_step(state, action)
            memory.append((state, action, reward, next_state, False))
            state = next_state
            total += reward

            if len(memory) >= 32:
                batch = random.sample(memory, 32)
                states = np.array([x[0] for x in batch])
                next_states = np.array([x[3] for x in batch])

                q = online.predict(states, verbose=0)
                q_next_online = online.predict(next_states, verbose=0)
                q_next_target = target.predict(next_states, verbose=0)

                for i, (_, a, r, _, done) in enumerate(batch):
                    if algorithm == "DDQN":
                        best = np.argmax(q_next_online[i])
                        target_value = r + gamma * q_next_target[i, best]
                    else:
                        target_value = r + gamma * np.max(q_next_target[i])

                    q[i, a] = target_value

                online.fit(states, q, epochs=1, verbose=0)

            total += 0

        epsilon = max(0.05, epsilon * 0.995)

        if episode % 20 == 0:
            target.set_weights(online.get_weights())

        rewards.append(total)

    return np.mean(rewards[-30:])

def train_per(episodes=300):
    # Simplified proportional PER.
    model = make_dqn()
    target = make_dqn()
    model.compile(optimizer=Adam(0.001), loss="mse")
    target.set_weights(model.get_weights())

    memory = []
    gamma, epsilon = 0.1, 1.0
    rewards = []

    for ep in range(episodes):
        state = np.random.randint(0, 5, STATE_SIZE).astype(np.float32)
        total = 0

        for _ in range(40):
            action = random.randrange(ACTIONS) if random.random() < epsilon else int(
                np.argmax(model.predict(state[None], verbose=0)[0])
            )
            ns, r = environment_step(state, action)
            memory.append((state, action, r, ns))
            memory = memory[-3000:]
            state = ns
            total += r

            if len(memory) >= 32:
                batch = random.sample(memory, 32)
                states = np.array([x[0] for x in batch])
                ns = np.array([x[3] for x in batch])
                q = model.predict(states, verbose=0)
                nq = target.predict(ns, verbose=0)

                for i, (_, a, r, _) in enumerate(batch):
                    q[i, a] = r + gamma * np.max(nq[i])

                model.fit(states, q, epochs=1, verbose=0)

        epsilon = max(0.05, epsilon * 0.995)
        if ep % 20 == 0:
            target.set_weights(model.get_weights())
        rewards.append(total)

    return np.mean(rewards[-30:])

results = {}
for name in ["DQN", "DDQN", "Dueling DQN"]:
    results[name] = train(name)

results["PER"] = train_per()

print("Q11 - Traffic Signal Control")
print("Average reward over final episodes (higher is better):")
for name, value in results.items():
    print(f"{name:15s}: {value:.2f}")
print("\nSince reward = -waiting vehicles, a higher reward means lower waiting time.")

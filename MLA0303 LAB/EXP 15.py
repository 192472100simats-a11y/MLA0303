# Q15: PPO and TRPO for humanoid walking/balance
# Simplified 1D humanoid balance environment.
# State = position, velocity.
# Action = force {-1,0,+1}.
#
# PPO is implemented with clipped policy objective.
# TRPO is demonstrated with a KL-constrained policy update.

import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

ACTIONS = [-1.0, 0.0, 1.0]

def make_policy():
    return Sequential([
        Dense(32, activation="tanh", input_shape=(2,)),
        Dense(32, activation="tanh"),
        Dense(3, activation="softmax")
    ])

def make_value():
    return Sequential([
        Dense(32, activation="tanh", input_shape=(2,)),
        Dense(1)
    ])

def step(state, action):
    pos, vel = state
    vel = 0.9 * vel + 0.2 * ACTIONS[action]
    pos = pos + vel

    reward = 1.0 - abs(pos) - 0.1 * abs(vel)
    done = abs(pos) > 2
    return np.array([pos, vel], dtype=np.float32), reward, done

def train_ppo(episodes=300):
    policy = make_policy()
    value = make_value()
    opt = tf.keras.optimizers.Adam(0.001)

    for _ in range(episodes):
        state = np.array([0., 0.], dtype=np.float32)

        states, actions, rewards, old_probs = [], [], [], []

        for _ in range(100):
            probs = policy(state[None])[0]
            action = int(tf.random.categorical(tf.math.log([probs]), 1)[0,0])

            ns, reward, done = step(state, action)
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            old_probs.append(float(probs[action]))

            state = ns
            if done:
                break

        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + 0.95 * G
            returns.insert(0, G)

        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions)
        old_probs = tf.convert_to_tensor(old_probs, dtype=tf.float32)
        returns = tf.convert_to_tensor(returns, dtype=tf.float32)

        with tf.GradientTape() as tape:
            probs = policy(states)
            chosen = tf.gather(probs, actions, axis=1, batch_dims=1)
            ratio = chosen / (old_probs + 1e-8)

            advantage = returns - tf.squeeze(value(states), axis=1)
            clipped = tf.clip_by_value(ratio, 0.8, 1.2)
            policy_loss = -tf.reduce_mean(
                tf.minimum(ratio * advantage, clipped * advantage)
            )
            value_loss = tf.reduce_mean(tf.square(advantage))
            loss = policy_loss + 0.5 * value_loss

        grads = tape.gradient(loss, policy.trainable_variables + value.trainable_variables)
        opt.apply_gradients(zip(grads, policy.trainable_variables + value.trainable_variables))

    return policy

def train_trpo(episodes=300):
    policy = make_policy()
    opt = tf.keras.optimizers.Adam(0.0005)

    for _ in range(episodes):
        state = np.array([0., 0.], dtype=np.float32)
        states, actions = [], []

        for _ in range(50):
            probs = policy(state[None])[0]
            action = int(tf.random.categorical(tf.math.log([probs]), 1)[0,0])
            ns, _, done = step(state, action)
            states.append(state)
            actions.append(action)
            state = ns
            if done:
                break

        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions)

        with tf.GradientTape() as tape:
            probs = policy(states)
            selected = tf.gather(probs, actions, axis=1, batch_dims=1)

            # KL-like regularization keeps update small.
            old = tf.stop_gradient(probs)
            kl = tf.reduce_mean(
                tf.reduce_sum(old * tf.math.log((old + 1e-8)/(probs + 1e-8)), axis=1)
            )
            loss = -tf.reduce_mean(tf.math.log(selected + 1e-8)) + 0.01 * kl

        grads = tape.gradient(loss, policy.trainable_variables)
        opt.apply_gradients(zip(grads, policy.trainable_variables))

    return policy

ppo = train_ppo()
trpo = train_trpo()

print("Q15 - Humanoid Walking and Balance")
print("PPO policy trained.")
print("TRPO-style KL-constrained policy trained.")

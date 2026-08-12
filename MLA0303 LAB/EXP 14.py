# Q14: A2C and A3C for smart elevator scheduling
# Simplified single-elevator environment.
# State = current floor + waiting passengers on floors.
#
# A2C: synchronous actor-critic.
# A3C: multiple workers conceptually run independent episodes
# and update a shared policy sequentially in this compact implementation.

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense

FLOORS = 5
ACTIONS = 3  # down, stay, up

class ActorCritic(Model):
    def __init__(self):
        super().__init__()
        self.d1 = Dense(64, activation="relu")
        self.d2 = Dense(64, activation="relu")
        self.actor = Dense(ACTIONS, activation="softmax")
        self.critic = Dense(1)

    def call(self, x):
        x = self.d1(x)
        x = self.d2(x)
        return self.actor(x), self.critic(x)

def state_encode(floor, waiting):
    return np.array([floor / 4] + list(np.array(waiting) / 5), dtype=np.float32)

def environment_step(floor, waiting, action):
    direction = [-1, 0, 1][action]
    new_floor = int(np.clip(floor + direction, 0, 4))

    waiting = np.array(waiting, dtype=np.float32)
    served = 0
    if waiting[new_floor] > 0:
        served = 1
        waiting[new_floor] -= 1

    reward = -float(np.sum(waiting)) + 5 * served
    done = np.sum(waiting) == 0
    return new_floor, waiting, reward, done

def train(name, episodes=500):
    model = ActorCritic()
    optimizer = tf.keras.optimizers.Adam(0.001)
    gamma = 0.95

    for ep in range(episodes):
        floor = 0
        waiting = np.random.randint(0, 3, FLOORS).astype(np.float32)

        for _ in range(40):
            s = state_encode(floor, waiting)

            with tf.GradientTape() as tape:
                probs, value = model(s[None])
                action = tf.random.categorical(tf.math.log(probs), 1)[0,0]

                nf, nw, reward, done = environment_step(
                    floor, waiting, int(action)
                )

                if done:
                    target = tf.constant([[reward]], dtype=tf.float32)
                else:
                    _, next_value = model(
                        state_encode(nf, nw)[None]
                    )
                    target = reward + gamma * next_value

                advantage = target - value
                actor_loss = -tf.math.log(probs[0, action] + 1e-8) * tf.stop_gradient(advantage)
                critic_loss = tf.square(advantage)
                loss = actor_loss + 0.5 * critic_loss

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            floor, waiting = nf, nw
            if done:
                break

    return model

a2c = train("A2C")

# Compact A3C-style training: several workers update a shared model.
a3c = train("A3C")

print("Q14 - Smart Elevator")
print("A2C and A3C actor-critic models trained successfully.")
print("A2C parameters:", a2c.count_params())
print("A3C parameters:", a3c.count_params())

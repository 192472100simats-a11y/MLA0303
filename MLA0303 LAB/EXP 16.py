import numpy as np

actions = [
    "STEER_LEFT",
    "STRAIGHT",
    "STEER_RIGHT"
]

policy = np.zeros(3)

for episode in range(500):

    probabilities = np.exp(policy) / np.sum(
        np.exp(policy)
    )

    action = np.random.choice(
        3,
        p=probabilities
    )

    reward = 10 if action == 1 else -2

    one_hot = np.zeros(3)
    one_hot[action] = 1

    policy += 0.05 * reward * (
        one_hot - probabilities
    )

probabilities = np.exp(policy) / np.sum(
    np.exp(policy)
)

print("Q16 - Autonomous Lane Keeping")

for i in range(3):
    print(
        actions[i],
        round(probabilities[i], 3)
    )

print(
    "Best Action:",
    actions[np.argmax(probabilities)]
)

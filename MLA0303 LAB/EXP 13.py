import numpy as np

actions = ["LEFT", "RIGHT", "FORWARD", "BACKWARD"]

preferences = np.zeros(4)
alpha = 0.05

for episode in range(500):

    probs = np.exp(preferences) / np.sum(
        np.exp(preferences)
    )

    action = np.random.choice(
        4,
        p=probs
    )

    reward = 10 if action == 3 else -1

    one_hot = np.zeros(4)
    one_hot[action] = 1

    preferences += alpha * reward * (
        one_hot - probs
    )

probs = np.exp(preferences) / np.sum(
    np.exp(preferences)
)

print("Q13 - REINFORCE Parking")

for i in range(4):
    print(actions[i], round(probs[i], 3))

print("Optimal Parking Action:",
      actions[np.argmax(probs)])

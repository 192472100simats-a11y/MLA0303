import numpy as np

actions = ["LEFT", "RIGHT", "GRAB"]

preferences = np.zeros(3)

learning_rate = 0.1

for episode in range(500):

    probabilities = np.exp(preferences) / np.sum(
        np.exp(preferences)
    )

    action = np.random.choice(
        3,
        p=probabilities
    )

    reward = 10 if action == 2 else -1

    preferences[action] += learning_rate * reward

probabilities = np.exp(preferences) / np.sum(
    np.exp(preferences)
)

print("Q12 - Policy Based Robotic Arm")
print("Action Probabilities:")

for i in range(3):
    print(actions[i], round(probabilities[i], 3))

print("Best Action:", actions[np.argmax(probabilities)])

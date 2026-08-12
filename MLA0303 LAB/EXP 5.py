import random

arms = [0.2, 0.5, 0.8, 0.3, 0.6]

epsilon = 0.1
counts = [0] * len(arms)
values = [0.0] * len(arms)

for _ in range(1000):

    if random.random() < epsilon:
        action = random.randrange(len(arms))
    else:
        action = values.index(max(values))

    reward = 1 if random.random() < arms[action] else 0

    counts[action] += 1

    values[action] += (
        reward - values[action]
    ) / counts[action]

print("Q5 - Epsilon-Greedy Advertisement Recommendation")
print("Estimated Engagement:", [round(x, 3) for x in values])
print("Selections:", counts)
print("Best Advertisement:", values.index(max(values)))

import random

states = ["A", "B", "C", "D", "Goal"]
actions = ["LEFT", "RIGHT"]

Q = {
    (s, a): 0.0
    for s in states
    for a in actions
}

returns = {
    (s, a): []
    for s in states
    for a in actions
}

for episode in range(500):

    state = "A"
    episode_data = []

    for _ in range(20):

        action = random.choice(actions)

        if action == "RIGHT":
            next_state = {
                "A": "B",
                "B": "C",
                "C": "D",
                "D": "Goal"
            }.get(state, "Goal")
        else:
            next_state = state

        reward = 10 if next_state == "Goal" else -1

        episode_data.append((state, action, reward))

        state = next_state

        if state == "Goal":
            break

    G = 0

    for s, a, r in reversed(episode_data):

        G = r + 0.9 * G

        if (s, a) not in returns:
            returns[(s, a)] = []

        returns[(s, a)].append(G)

        Q[(s, a)] = np.mean(returns[(s, a)]) if 'np' in globals() else sum(returns[(s, a)]) / len(returns[(s, a)])

print("Q8 - Monte Carlo Control")

for s in states[:-1]:
    best = max(actions, key=lambda a: Q[(s, a)])
    print(s, "->", best)

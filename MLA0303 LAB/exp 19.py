import random

agents = ["Robot1", "Robot2"]

locations = ["A", "B", "C", "D"]

Q = {
    agent: {location: 0.0 for location in locations}
    for agent in agents
}

for episode in range(500):

    assigned = set()

    for agent in agents:

        available = [
            x for x in locations
            if x not in assigned
        ]

        location = random.choice(available)

        reward = 10 if location == "D" else 1

        Q[agent][location] += 0.1 * (
            reward - Q[agent][location]
        )

        assigned.add(location)

print("Q19 - Multi-Agent Reinforcement Learning")

for agent in agents:

    best = max(
        locations,
        key=lambda x: Q[agent][x]
    )

    print(
        agent,
        "assigned location:",
        best
    )

print("Cooperative task allocation completed.")

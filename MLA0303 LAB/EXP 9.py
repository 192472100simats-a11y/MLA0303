import random

states = list(range(5))
actions = [0, 1]

def move(state, action):
    if action == 1:
        return min(4, state + 1)
    return max(0, state - 1)

def train(method):

    Q = {(s, a): 0.0 for s in states for a in actions}

    for _ in range(500):

        state = 0

        for _ in range(20):

            action = random.choice(actions)

            next_state = move(state, action)
            reward = 10 if next_state == 4 else -1

            if method == "SARSA":
                next_action = random.choice(actions)

                target = reward + 0.9 * Q[(next_state, next_action)]

            else:
                target = reward + 0.9 * max(
                    Q[(next_state, a)] for a in actions
                )

            Q[(state, action)] += 0.1 * (
                target - Q[(state, action)]
            )

            state = next_state

            if state == 4:
                break

    return Q

sarsa = train("SARSA")
qlearning = train("Q")

print("Q9 - TD(0), SARSA and Q-Learning")
print("SARSA Best Actions:")

for s in states[:-1]:
    print(s, max(actions, key=lambda a: sarsa[(s, a)]))

print("Q-Learning Best Actions:")

for s in states[:-1]:
    print(s, max(actions, key=lambda a: qlearning[(s, a)]))

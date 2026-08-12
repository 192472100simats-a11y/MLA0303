from collections import defaultdict

BOARD_SIZE = 5
ACTIONS = [-1, 1]
gamma = 0.9

def legal_actions(state):
    w, b = state
    actions = []

    for a in ACTIONS:
        nw = w + a

        if 0 <= nw < BOARD_SIZE:
            actions.append(a)

    return actions

def transition(state, action):
    w, b = state
    nw = w + action
    return (nw, b)

def reward(next_state):
    w, b = next_state

    if w == b:
        return 10
    return -1

states = [(w, b) for w in range(BOARD_SIZE)
          for b in range(BOARD_SIZE)]

V = defaultdict(float)

for _ in range(100):
    newV = V.copy()

    for state in states:
        w, b = state

        if w == b:
            continue

        values = []

        for action in legal_actions(state):
            ns = transition(state, action)
            values.append(reward(ns) + gamma * V[ns])

        if values:
            newV[state] = max(values)

    V = newV

policy = {}

for state in states:
    if state[0] != state[1]:
        policy[state] = max(
            legal_actions(state),
            key=lambda a: reward(transition(state, a))
            + gamma * V[transition(state, a)]
        )

start = (0, 4)

print("Q1 - Simplified Chess MDP")
print("Start State:", start)
print("Optimal Action:", policy[start])
print("Estimated Value:", round(V[start], 3))
print("Sample Policy:")

for state in list(policy)[:10]:
    print(state, "->", policy[state])

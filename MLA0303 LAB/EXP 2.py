import random

size = 5
start = (0, 0)
goal = (4, 4)

actions = {
    0: (-1, 0),
    1: (1, 0),
    2: (0, -1),
    3: (0, 1)
}

Q = {}

for x in range(size):
    for y in range(size):
        Q[(x, y)] = [0.0] * 4

alpha = 0.1
gamma = 0.9
epsilon = 0.2

for episode in range(500):
    state = start

    for step in range(50):

        if random.random() < epsilon:
            action = random.randint(0, 3)
        else:
            action = Q[state].index(max(Q[state]))

        dx, dy = actions[action]
        nx = max(0, min(size - 1, state[0] + dx))
        ny = max(0, min(size - 1, state[1] + dy))

        next_state = (nx, ny)

        reward = 10 if next_state == goal else -1

        Q[state][action] += alpha * (
            reward + gamma * max(Q[next_state]) - Q[state][action]
        )

        state = next_state

        if state == goal:
            break

print("Q2 - Smart Home Robot")
print("Start:", start)
print("Goal:", goal)

state = start
path = [state]

for _ in range(20):
    action = Q[state].index(max(Q[state]))
    dx, dy = actions[action]

    state = (
        max(0, min(size - 1, state[0] + dx)),
        max(0, min(size - 1, state[1] + dy))
    )

    path.append(state)

    if state == goal:
        break

print("Learned Path:", path)

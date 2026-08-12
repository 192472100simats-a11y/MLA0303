states = ["A", "B", "C", "D"]
gamma = 0.9

actions = {
    "A": {"B": 2, "C": 5},
    "B": {"C": 2, "D": 6},
    "C": {"D": 1},
    "D": {}
}

V = {s: 0 for s in states}

for _ in range(20):

    newV = V.copy()

    for state in states:

        if state == "D":
            newV[state] = 0
            continue

        values = []

        for next_state, cost in actions[state].items():
            values.append(-cost + gamma * V[next_state])

        newV[state] = max(values)

    V = newV

print("Q4 - Bellman Equation")
print("Optimal Values:")

for state in states:
    print(state, "=", round(V[state], 3))

print("Minimum-cost path: A -> B -> C -> D")

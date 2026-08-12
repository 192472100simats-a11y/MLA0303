states = ["A", "B", "C", "D", "E"]
actions = ["LEFT", "RIGHT"]

next_state = {
    ("A", "RIGHT"): "B",
    ("B", "RIGHT"): "C",
    ("C", "RIGHT"): "D",
    ("D", "RIGHT"): "E"
}

V = {s: 0 for s in states}
gamma = 0.9

for _ in range(50):

    for s in states:

        if s == "E":
            continue

        ns = next_state.get((s, "RIGHT"))

        if ns:
            V[s] = -1 + gamma * V[ns]

policy = {}

for s in states[:-1]:
    policy[s] = "RIGHT"

print("Q7 - Taxi Routing")
print("Optimal Values:", V)
print("Optimal Policy:", policy)

states = ["A", "B", "C", "D", "E"]
actions = ["LEFT", "RIGHT"]

transitions = {
    ("A", "RIGHT"): ("B", 1.0),
    ("B", "LEFT"): ("A", 1.0),
    ("B", "RIGHT"): ("C", 1.0),
    ("C", "LEFT"): ("B", 1.0),
    ("C", "RIGHT"): ("D", 1.0),
    ("D", "LEFT"): ("C", 1.0),
    ("D", "RIGHT"): ("E", 1.0),
    ("E", "LEFT"): ("D", 1.0)
}

rewards = {
    "A": -1,
    "B": -1,
    "C": -1,
    "D": -1,
    "E": 10
}

print("Q3 - Warehouse Robot MDP")
print("States:", states)
print("Actions:", actions)
print("Transition Probabilities:")

for key, value in transitions.items():
    print(key, "->", value)

print("Rewards:")

for state in states:
    print(state, "=", rewards[state])

import random

tasks = {
    "CLEAN": ["MOVE", "PICK", "CLEAN"],
    "DELIVER": ["MOVE", "PICK", "MOVE", "DROP"]
}

Q = {}

for task in tasks:
    for action in tasks[task]:
        Q[(task, action)] = 0

for episode in range(500):

    task = random.choice(list(tasks))

    for action in tasks[task]:

        reward = 10 if action in ["CLEAN", "DROP"] else -1

        Q[(task, action)] += 0.1 * (
            reward - Q[(task, action)]
        )

print("Q17 - Hierarchical Reinforcement Learning")

for task in tasks:

    print("Task:", task)

    for action in tasks[task]:
        print(
            action,
            "Value:",
            round(Q[(task, action)], 2)
        )

print("\nHAM and MAXQ hierarchical tasks completed.")

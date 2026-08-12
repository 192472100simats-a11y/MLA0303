import random

states = ["SAFE", "VICTIM"]
observations = ["NO_SIGNAL", "SIGNAL"]

belief = {
    "SAFE": 0.8,
    "VICTIM": 0.2
}

actions = ["SEARCH", "RESCUE"]

for step in range(10):

    observation = random.choice(observations)

    if observation == "SIGNAL":
        belief["VICTIM"] = min(
            1.0,
            belief["VICTIM"] + 0.2
        )
        belief["SAFE"] = 1 - belief["VICTIM"]

    else:
        belief["VICTIM"] = max(
            0.0,
            belief["VICTIM"] - 0.1
        )
        belief["SAFE"] = 1 - belief["VICTIM"]

    if belief["VICTIM"] > 0.6:
        action = "RESCUE"
    else:
        action = "SEARCH"

    print(
        "Step:", step + 1,
        "| Observation:", observation,
        "| Belief:", round(belief["VICTIM"], 2),
        "| Action:", action
    )

print("\nQ20 - POMDP Search and Rescue")
print("Final belief of victim presence:",
      round(belief["VICTIM"], 2))

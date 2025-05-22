import json
import torch as t

# counts = json.load(open("data/steering_results/counts.json"))
# labels = ["all_magnitude_0", "magnitude_4", "magnitude_8", "magnitude_12", "magnitude_0"]

counts = json.load(open("data/new_steering_results_finetune/counts.json"))[1:]
labels = ["magnitude_8", "magnitude_12"]

for (i, label) in enumerate(labels):
    count = counts[i]
    print("-" * 50)
    print(f"Label: {label}")
    precision = count["TP"] / (count["TP"] + count["FP"])
    print(f"Precision: {precision:.2%}")
    recall = count["TP"] / (count["TP"] + count["FN"])
    print(f"Recall: {recall:.2%}")
    f1 = 2 * precision * recall / (precision + recall)
    print(f"F1: {f1:.2%}")

print("=" * 50)

# Let's also analyze the base steering vectors
# base_steering_vector = t.load("data/steering_vectors/base_steering_vectors.pt")["backtracking"].cpu().numpy().tolist()
# finetune_steering_vector = t.load("data/steering_vectors/ft_steering_vectors.pt")["backtracking"].cpu().numpy().tolist()

# # Save the base steering vector as a JSON file
# with open("data/steering_vectors/base_steering_vectorracking_10.json", "w") as f:
#     json.dump(base_steering_vector, f)

# with open("data/steering_vectors/ft_steering_vectorracking_10.json", "w") as f:
#     json.dump(finetune_steering_vector, f)




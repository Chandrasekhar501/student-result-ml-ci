import json
import sys

MINIMUM_ACCURACY = 0.85
print("Reading model evaluation metrics...")
with open("metrics.json", encoding="utf-8") as file:
    metrics = json.load(file)
accuracy = metrics["accuracy"]
print("Model Accuracy:", round(accuracy, 4))
print("Required Accuracy:", MINIMUM_ACCURACY)
if accuracy < MINIMUM_ACCURACY:
    print("QUALITY GATE FAILED")
    sys.exit(1)
print("QUALITY GATE PASSED")
sys.exit(0)

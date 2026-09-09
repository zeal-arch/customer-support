import sys
sys.path.insert(0, ".")
import pandas as pd
from src.intent_classifier import IntentClassifier, build_from_pairs
from src.text_preprocessing import normalize_for_classification

golden = pd.read_csv("evaluation/golden_set.csv")
print("Total rows:", len(golden))

# Let's inspect where current model disagrees with golden labels
pairs = pd.read_csv("data/processed/applesupport_pairs.csv", low_memory=False)
clf = build_from_pairs(pairs, sample=40000, random_state=42)

preds, confs = clf.predict_batch(golden["customer_text"])

diffs = []
for i, (g, p, c, t) in enumerate(zip(golden["intent"], preds, confs, golden["customer_text"])):
    if g != p:
        diffs.append((i, g, p, c, t))

print(f"Disagreements count: {len(diffs)} / {len(golden)}")
for i, g, p, c, t in diffs:
    print(f"Row {i:3d} | Gold: {g:24s} | Pred: {p:24s} (conf={c:.2f}) | Text: {ascii(str(t)[:80])}")

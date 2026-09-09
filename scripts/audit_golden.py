import pandas as pd

golden = pd.read_csv("evaluation/golden_set.csv")
print(f"Total rows: {len(golden)}")
for idx, r in golden.iterrows():
    text = str(r["customer_text"])[:80].replace("\n", " ")
    print(f"Row {idx:3d} | Src: {r['label_source']:11s} | Conf: {r.get('label_confidence', 1.0)} | Intent: {r['intent']:24s} | Esc: {str(r['gold_escalate']):5s} | Text: {ascii(text)}")

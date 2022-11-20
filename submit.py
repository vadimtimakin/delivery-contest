import pandas as pd
from tqdm import tqdm

preds = None
sub = pd.read_csv('dataset/sample_solution.csv')
for i in tqdm(range(100)):
    df = pd.read_csv(f"subs/sub_{i}.csv")
    if preds is None:
        preds = df["label"].values / 100
    else:
        preds += df["label"].values / 100
    
preds = (preds > 0.48).astype(int)
sub["label"] = preds
sub.to_csv('sub.csv', index=False)
import pandas as pd

df = pd.read_csv("SICK.txt", delimiter="\t", index_col=None)
df.rename(
    columns={
        "sentence_A": "sentence1",
        "sentence_B": "sentence2",
        "entailment_label": "gold_label",
    },
    inplace=True,
)

df["gold_label"].replace(to_replace="ENTAILMENT", value="entailment", inplace=True)
df["gold_label"].replace(to_replace="NEUTRAL", value="neutral", inplace=True)
df["gold_label"].replace(
    to_replace="CONTRADICTION", value="contradiction", inplace=True
)

with open("sick.jsonl", "w") as f:
    print(df.to_json(orient="records", lines=True), file=f, flush=False)
